use rand::prelude::*;
use rand_distr::{Normal, Poisson, StandardNormal, Uniform};
use rayon::prelude::*;

macro_rules! time {
    ($func:expr) => {{
        let start = std::time::Instant::now();
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
pub struct EuropeanOptionCEV {
    pub S: f64,    // stock price
    pub K: f64,    // strike price
    pub r: f64,    // risk-free interest rate
    pub T: f64,    // time to expiration
    pub sig: f64,  // volatility
    pub beta: f64, // beta
    pub option_type: OptionType,
}

impl EuropeanOptionCEV {
    pub fn new(
        S: f64,
        K: f64,
        r: f64,
        T: f64,
        sig: f64,
        beta: f64,
        option_type: OptionType,
    ) -> Self {
        Self {
            S,
            K,
            r,
            T,
            sig,
            beta,
            option_type,
        }
    }
}

#[derive(Clone, Copy)]
pub struct MonteCarloCEV {
    pub n: usize, // number of time steps
    pub m: usize, // number of simulated paths
    pub is_antithetic: bool,
}

impl MonteCarloCEV {
    pub fn new(n: usize, m: usize, is_antithetic: bool) -> Self {
        Self {
            n,
            m,
            is_antithetic,
        }
    }
}

#[derive(Clone, Copy)]
pub struct JumpProcess {
    pub mu: f64,     // drift
    pub sigma: f64,  // volatility
    pub lambda: f64, // jump intensity
    pub compensator: f64,
    pub jump_distribution: Normal<f64>,
}

impl JumpProcess {
    pub fn new(mu: f64, sigma: f64, lambda: f64) -> Self {
        let compensator = (lambda * (mu + 0.5 * sigma.powi(2))).exp() - 1.0;
        let jump_distribution = Normal::new(mu, sigma).unwrap();
        Self {
            mu,
            sigma,
            lambda,
            compensator,
            jump_distribution,
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

pub struct MonteCarloEuropeanOptionCEVJumps {
    pub option: EuropeanOptionCEV,
    pub mc: MonteCarloCEV,
    pub jump: JumpProcess,
    pub dt: f64,
    pub lnS: f64,
}

impl MonteCarloEuropeanOptionCEVJumps {
    pub fn new(option: EuropeanOptionCEV, mc: MonteCarloCEV, jump: JumpProcess) -> Self {
        let dt = option.T / mc.n as f64;
        let lnS = option.S.ln();

        Self {
            option,
            mc,
            jump,
            dt,
            lnS,
        }
    }

    fn simulate(&self) -> Vec<f64> {
        // Precompute constants
        let lnS = self.lnS;

        // Use SmallRng for faster RNG
        let master_seed = 45u64;
        let mut master_rng = SmallRng::seed_from_u64(master_seed);
        let seeds: Vec<u64> = (0..self.mc.m).map(|_| master_rng.gen()).collect();

        let uniform = Uniform::new(0.0, self.option.T).unwrap();
        let poisson = Poisson::new(self.dt / self.jump.lambda).unwrap();

        let mut C_T = vec![0.0; self.mc.m];

        C_T.par_iter_mut().zip(seeds).for_each(|(val, seed)| {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut lnS1 = lnS;
            let mut lnS2 = lnS;

            // jump component
            let num_jumps: usize = rng.sample(poisson) as usize;
            let mut jump_times: Vec<f64> = (0..num_jumps).map(|_| rng.sample(uniform)).collect();
            jump_times.sort_by(f64::total_cmp);
            if num_jumps > 8 {
                println!("Number of jumps: {num_jumps}");
            }
            let mut curr_jump = 0;

            for i in 0..self.mc.n {
                let t = i as f64 * self.dt;
                let epsilon: f64 = rng.sample(StandardNormal);

                let S1 = lnS1.exp();
                let vol1 = self.option.sig * S1.powf((self.option.beta - 2.0) / 2.0);
                let drift1 = (self.option.r
                    - 0.5 * self.option.sig.powi(2) * S1.powf(self.option.beta - 2.0))
                * self.dt;
                let diffusion1 = vol1 * self.dt.sqrt() * epsilon;

                // jump component
                lnS1 += drift1 + diffusion1;
                while curr_jump < num_jumps && jump_times[curr_jump] - t < self.dt {
                    let jump: f64 = rng.sample(self.jump.jump_distribution);
                    lnS1 += jump;
                    curr_jump += 1;
                }

                // antithetical
                if self.mc.is_antithetic {
                    let S2 = lnS2.exp();
                    let vol2 = self.option.sig * S2.powf((self.option.beta - 2.0) / 2.0);
                    let drift2 = (self.option.r + self.jump.compensator
                        - 0.5 * self.option.sig.powi(2) * S2.powf(self.option.beta - 2.0))
                        * self.dt;
                    let diffusion2 = vol2 * self.dt.sqrt() * -epsilon;

                    lnS2 += drift2 + diffusion2;

                    while curr_jump < num_jumps
                        && jump_times[curr_jump] - t < self.dt
                    {
                        let jump: f64 = rng.sample(self.jump.jump_distribution);
                        lnS2 += jump;
                        curr_jump += 1;
                    }
                }
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

impl std::fmt::Display for MonteCarloStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "C0: {:.4}, SD: {:.4}, SE: {:.4}",
            self.C0, self.SD, self.SE
        )
    }
}

pub fn a() {
    let S = 100.0;
    let K = 100.0;
    let r = 0.001;
    let T = 2.0 / 12.0;
    let sig = 0.3;
    let beta = 1.2;
    let option_type = OptionType::Put;
    let option = EuropeanOptionCEV::new(S, K, r, T, sig, beta, option_type);
    let n = 300;
    let m = 1_000_000;
    let is_antithetic = false;
    let mc = MonteCarloCEV::new(n, m, is_antithetic);
    let jump = JumpProcess::new(-0.01, 0.09, 1.0 / 52.0);
    let mc_option = MonteCarloEuropeanOptionCEVJumps::new(option, mc, jump);
    let (C_T, duration) = time!(mc_option.simulate());
    let stats = mc_option.option_statistics(C_T);
    println!("Monte Carlo Simulation took: {:?}", duration);
    println!("Option Price: {}", stats.C0);
    println!("Standard Deviation: {}", stats.SD);
    println!("Standard Error: {}", stats.SE);

    let z = 1.96; // 95% confidence level
    let lower_bound = stats.C0 - z * stats.SE;
    let upper_bound = stats.C0 + z * stats.SE;
    println!(
        "95% Confidence Interval: [{:.4}, {:.4}]",
        lower_bound, upper_bound
    );

    let call_option_type = OptionType::Call;
    let call_option = EuropeanOptionCEV::new(S, K, r, T, sig, beta, call_option_type);
    let mc_call_option = MonteCarloEuropeanOptionCEVJumps::new(call_option, mc, jump);
    let (call_C_T, call_duration) = time!(mc_call_option.simulate());
    let call_stats = mc_call_option.option_statistics(call_C_T);
    println!("Call Option Monte Carlo Simulation took: {:?}", call_duration);
    println!("Call Option Price: {}", call_stats.C0);
    println!("Call Option Standard Deviation: {}", call_stats.SD);
    println!("Call Option Standard Error: {}", call_stats.SE);

    let call_lower_bound = call_stats.C0 - z * call_stats.SE;
    let call_upper_bound = call_stats.C0 + z * call_stats.SE;
    println!(
        "Call Option 95% Confidence Interval: [{:.4}, {:.4}]",
        call_lower_bound, call_upper_bound
    );

    // Verify Put-Call Parity
    let put_call_parity_diff = (call_stats.C0 - stats.C0) - (S - K * (-r * T).exp());
    println!(
        "Put-Call Parity Difference: {:.4} (should be close to 0)",
        put_call_parity_diff
    );
}
