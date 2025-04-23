use nalgebra::par_iter::ParColumnIter;
use nalgebra::DMatrix;
use plotly::common::Mode;
use plotly::Plot;
use plotly::Scatter;
use polars::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::time::Instant;

macro_rules! time {
    ($func:expr) => {{
        let start = Instant::now();
        let result = $func;
        let duration = start.elapsed();
        (result, duration)
    }};
}

#[derive(Debug, Clone, Copy)]
enum OptionType {
    Call,
    Put,
}

#[derive(Clone, Copy)]
pub struct EuropeanOption {
    pub S: f64,   // stock price
    pub K: f64,   // strike price
    pub r: f64,   // risk-free interest rate
    pub T: f64,   // time to expiration
    pub sig: f64, // volatility
    pub div: f64, // dividend yield
    pub option_type: OptionType,
}

impl EuropeanOption {
    pub fn new(
        S: f64,
        K: f64,
        r: f64,
        T: f64,
        sig: f64,
        div: f64,
        option_type: OptionType,
    ) -> Self {
        Self {
            S,
            K,
            r,
            T,
            sig,
            div,
            option_type,
        }
    }

    pub fn black_scholes(&self) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let d1 = (self.S.ln() - self.K.ln()
            + (self.r - self.div + 0.5 * self.sig.powi(2)) * self.T)
            / (self.sig * self.T.sqrt());
        let d2 = d1 - self.sig * self.T.sqrt();

        match self.option_type {
            OptionType::Call => {
                self.S * (-self.div * self.T).exp() * normal.cdf(d1)
                    - self.K * (-self.r * self.T).exp() * normal.cdf(d2)
            }
            OptionType::Put => {
                self.K * (-self.r * self.T).exp() * normal.cdf(-d2)
                    - self.S * (-self.div * self.T).exp() * normal.cdf(-d1)
            }
        }
    }
}

pub struct MonteCarlo {
    pub n: usize, // number of time steps
    pub m: usize, // number of simulated paths
    pub is_antithetic: bool,
    pub is_control: bool,
}

impl MonteCarlo {
    pub fn new(n: usize, m: usize, is_antithetic: bool, is_control: bool) -> Self {
        Self {
            n,
            m,
            is_antithetic,
            is_control,
        }
    }
}

pub struct MonteCarloStatistics {
    pub sum_CT: f64,
    pub sum_CT2: f64,
    pub C0: f64,
    pub SD: f64,
    pub SE: f64,
}

pub struct MonteCarloEuropeanOption {
    pub option: EuropeanOption,
    pub mc: MonteCarlo,
    pub dt: f64,
    pub nudt: f64,
    pub sigsdt: f64,
    pub lnS: f64,
    pub erddt: f64,
}

impl MonteCarloEuropeanOption {
    pub fn new(option: EuropeanOption, mc: MonteCarlo) -> Self {
        let dt = option.T / mc.n as f64;
        let nudt = (option.r - option.div - 0.5 * option.sig.powi(2)) * dt;
        let sigsdt = option.sig * dt.sqrt();
        let lnS = option.S.ln();
        let erddt = ((option.r - option.div) * dt).exp();

        Self {
            option,
            mc,
            dt,
            nudt,
            sigsdt,
            lnS,
            erddt,
        }
    }

    fn delta(&self, S: f64) -> f64 {
        let d1 = (S.ln() - self.option.K.ln()
            + (self.option.r - self.option.div + 0.5 * self.option.sig.powi(2)) * self.option.T)
            / (self.option.sig * self.option.T.sqrt());
        let normal = Normal::new(0.0, 1.0).unwrap();
        match self.option.option_type {
            OptionType::Call => normal.cdf(d1),
            OptionType::Put => normal.cdf(d1) - 1.0,
        }
    }

    fn simulate(&self) -> Vec<f64> {
        // Precompute constants
        let nudt = self.nudt;
        let sigsdt = self.sigsdt;
        let lnS = self.lnS;

        // Use SmallRng for faster RNG
        let master_seed = 45u64;
        let mut master_rng = SmallRng::seed_from_u64(master_seed);
        let seeds: Vec<u64> = (0..self.mc.m).map(|_| master_rng.gen()).collect();

        let mut C_T = vec![0.0; self.mc.m];

        C_T.par_iter_mut().zip(seeds).for_each(|(val, seed)| {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut lnS1 = lnS;
            let mut lnS2 = lnS;

            let mut St1 = self.option.S;
            let mut St2 = self.option.S;
            let mut cv1 = 0.0;
            let mut cv2 = 0.0;

            for i in 0..self.mc.n {
                let t = (i - 1) as f64 * self.dt;
                let delta1 = self.delta(St1);
                let delta2 = self.delta(St2);
                let epsilon: f64 = rng.sample(StandardNormal);

                lnS1 += nudt + sigsdt * epsilon;
                if self.mc.is_antithetic {
                    lnS2 += nudt - sigsdt * epsilon;
                }

                let Stn1 = lnS1.exp();
                let Stn2 = lnS2.exp();
                
            }

            let payoff = |lnS: f64| match self.option.option_type {
                OptionType::Call => (lnS.exp() - self.option.K).max(0.0),
                OptionType::Put => (self.option.K - lnS.exp()).max(0.0),
            };

            *val = if self.mc.is_antithetic {
                0.5 * (payoff(lnS1) + payoff(lnS2))
            } else {
                payoff(lnS1)
            };
        });

        C_T
    }

    fn simulate_closed_form(&self) -> Vec<f64> {
        // Precompute constants for closed-form lognormal model
        // Should be (mu - 0.5 * sigma^2) * T
        // Should be sigma * sqrt(T)
        let lnS = self.lnS;

        // Generate master RNG
        let master_seed = 42u64;
        let mut master_rng = SmallRng::seed_from_u64(master_seed);
        let seeds: Vec<u64> = (0..self.mc.m).map(|_| master_rng.gen()).collect();

        // Generate S_T directly in parallel
        seeds
            .par_iter()
            .map(|&seed| {
                let mut rng = SmallRng::seed_from_u64(seed);
                let z: f64 = rng.sample(StandardNormal);
                let n = self.mc.n as f64;
                let S = (lnS + self.nudt * n + self.sigsdt * z * n.sqrt()).exp();

                match self.option.option_type {
                    OptionType::Call => (S - self.option.K).max(0.0),
                    OptionType::Put => (self.option.K - S).max(0.0),
                }
            })
            .collect()
    }

    pub fn option_statistics(&self, C_T: Vec<f64>) -> MonteCarloStatistics {
        let sum_CT = C_T.iter().sum::<f64>();
        let sum_CT2 = C_T.iter().map(|x| x.powi(2)).sum::<f64>();
        let C0 = sum_CT / self.mc.m as f64 * (-self.option.r * self.option.T).exp();
        let SD = (((sum_CT2 - sum_CT.powi(2) / self.mc.m as f64)
            * (-2.0 * self.option.r * self.option.T).exp())
            / (self.mc.m as f64 - 1.0))
            .sqrt();
        let SE = SD / (self.mc.m as f64).sqrt();
        MonteCarloStatistics {
            sum_CT,
            sum_CT2,
            C0,
            SD,
            SE,
        }
    }
}

impl std::fmt::Display for MonteCarloEuropeanOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Monte Carlo European Option: \n\
            S: {}, K: {}, r: {}, T: {}, sig: {}, div: {}, option_type: {:?}, \n\
            n: {}, m: {}",
            self.option.S,
            self.option.K,
            self.option.r,
            self.option.T,
            self.option.sig,
            self.option.div,
            self.option.option_type,
            self.mc.n,
            self.mc.m
        )
    }
}

pub fn a() -> PolarsResult<()> {
    // option parameters
    let r = 0.01;
    let div = 0.005;
    let sig: f64 = 0.4;
    let S: f64 = 100.0;
    let K = 100.0;
    let T = 1.0;
    let option_type = OptionType::Call;

    // monte carlo parameters
    let n = 700; //300; //-700 time steps
    let m = 25_000_000; //1_000_000; //-5_000_000 simulated paths

    let option = EuropeanOption::new(S, K, r, T, sig, div, option_type);
    let mc = MonteCarlo::new(n, m, false, false);
    let mc_option = MonteCarloEuropeanOption::new(option, mc);

    let (option_prices, duration_simulation) = time!(mc_option.simulate());
    println!("Option Prices (Time: {:?})", duration_simulation);

    let (stats, duration_stats) = time!(mc_option.option_statistics(option_prices));
    println!(
        "Monte Carlo Statistics:\nC0: {:.4}, SD: {:.4}, SE: {:.4} (Time: {:?})",
        stats.C0, stats.SD, stats.SE, duration_stats
    );

    // Closed-form simulation
    let (option_prices_cf, duration_simulation_cf) = time!(mc_option.simulate_closed_form());

    println!(
        "Option Prices (Closed Form) (Time: {:?})",
        duration_simulation_cf
    );

    let (stats_cf, duration_stats_cf) = time!(mc_option.option_statistics(option_prices_cf));
    println!(
        "Monte Carlo Statistics (Closed Form):\nC0: {:.4}, SD: {:.4}, SE: {:.4} (Time: {:?})",
        stats_cf.C0, stats_cf.SD, stats_cf.SE, duration_stats_cf
    );
    let bs_price = option.black_scholes();
    println!("Black-Scholes Price: {:.4}", bs_price);

    // let paths = mc_option.paths();
    // let plot = mc_option.plot_paths(paths);
    // plot.show();
    Ok(())
}

pub fn b() {
    // option parameters
    let r = 0.01;
    let div = 0.005;
    let sig: f64 = 0.4;
    let S: f64 = 100.0;
    let K = 100.0;
    let T = 1.0;
    let option_type = OptionType::Call;

    let option = EuropeanOption::new(S, K, r, T, sig, div, option_type);
    let n = 700;
    let m = 5_000_000;
    let mc = MonteCarlo::new(n, m, false, false);
    let mc_option = MonteCarloEuropeanOption::new(option, mc);

    // Simulate stock prices
    let (option_prices, duration_simulation) = time!(mc_option.simulate());
    println!("Option Prices (Time: {:?})", duration_simulation);

    // Calculate statistics
    let (stats, duration_stats) = time!(mc_option.option_statistics(option_prices));
    println!(
        "Monte Carlo Statistics:\nC0: {:.4}, SD: {:.4}, SE: {:.4} (Time: {:?})",
        stats.C0, stats.SD, stats.SE, duration_stats
    );

    // Now with antithetic
    let mc_a = MonteCarlo::new(n, m, true, false);
    let mc_option_a = MonteCarloEuropeanOption::new(option, mc_a);

    let (option_prices, duration_simulation) = time!(mc_option_a.simulate());
    println!(
        "Option Prices with Antithetic (Time: {:?})",
        duration_simulation
    );

    let (stats, duration_stats) = time!(mc_option_a.option_statistics(option_prices));
    println!(
        "Monte Carlo Statistics with Antithetic:\nC0: {:.4}, SD: {:.4}, SE: {:.4} (Time: {:?})",
        stats.C0, stats.SD, stats.SE, duration_stats
    );

    let bs_price = option.black_scholes();
    println!("Black-Scholes Price: {:.4}", bs_price);
}
