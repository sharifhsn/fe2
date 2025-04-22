use nalgebra::par_iter::ParColumnIter;
use nalgebra::DMatrix;
use plotly::common::Mode;
use plotly::Plot;
use plotly::Scatter;
use polars::prelude::*;
use rand::{prelude::*, rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
enum OptionType {
    Call,
    Put,
}

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
}

pub struct MonteCarlo {
    pub n: usize, // number of time steps
    pub m: usize, // number of simulated paths
}

impl MonteCarlo {
    pub fn new(n: usize, m: usize) -> Self {
        Self { n, m }
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
}

impl MonteCarloEuropeanOption {
    pub fn new(option: EuropeanOption, mc: MonteCarlo) -> Self {
        let dt = option.T / mc.n as f64;
        let nudt = (option.r - option.div - 0.5 * option.sig.powi(2)) * dt;
        let sigsdt = option.sig * dt.sqrt();
        let lnS = option.S.ln();

        Self {
            option,
            mc,
            dt,
            nudt,
            sigsdt,
            lnS,
        }
    }

    fn simulate_stock(&self) -> Vec<f64> {
        let mut S_T: Vec<f64> = vec![self.lnS; self.mc.m];
        S_T.par_iter_mut().for_each(|val| {
            let mut rng = rand::thread_rng();
            for _ in 0..self.mc.n {
                let epsilon: f64 = rng.sample(StandardNormal);
                *val += self.nudt + self.sigsdt * epsilon;
            }
        });

        let S_T = S_T.par_iter().map(|val| val.exp()).collect::<Vec<f64>>();
        S_T
    }

    fn calculate_options(&self, S_T: Vec<f64>) -> Vec<f64> {
        // calculate the option price
        let C_T = S_T
            .par_iter()
            .map(|S_T| {
                // let S_T = val.exp();
                match self.option.option_type {
                    OptionType::Call => (S_T - self.option.K).max(0.0),
                    OptionType::Put => (self.option.K - S_T).max(0.0),
                }
            })
            .collect::<Vec<f64>>();
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

    fn paths(&self) -> DMatrix<f64> {
        let mut mat: DMatrix<f64> = DMatrix::zeros(self.mc.n, self.mc.m);
        mat.row_mut(0).fill(self.lnS);
        mat.par_column_iter_mut().for_each(|mut col| {
            let mut rng = rand::thread_rng();
            for i in 1..self.mc.n {
                let epsilon: f64 = rng.sample(StandardNormal);
                col[i] = col[i - 1] + self.nudt + self.sigsdt * epsilon;
            }
        });
        mat
    }
    fn plot_paths(&self, paths: DMatrix<f64>) -> Plot {
        let mut plot = Plot::new();
        for col in paths.column_iter() {
            let path: Vec<f64> = col.iter().copied().collect();
            let time: Vec<f64> = (0..self.mc.n).map(|i| i as f64 * self.dt).collect();

            let trace = Scatter::new(time, path).mode(Mode::Lines);
            plot.add_trace(trace);
        }
        plot
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
    let r = 0.06; //0.01;
    let div = 0.03; //0.005;
    let sig: f64 = 0.2; //0.40;
    let S: f64 = 100.0;
    let K = 100.0;
    let T = 1.0;
    let option_type = OptionType::Call;

    // monte carlo parameters
    let n = 10; //300; //-700 time steps
    let m = 100; //1_000_000; //-5_000_000 simulated paths

    // derived parameters
    let dt = T / n as f64;
    // println!("dt: {}", dt);
    let nudt = (r - div - 0.5 * sig.powi(2)) * dt;
    // println!("nudt: {}", nudt);
    let sigsdt = sig * dt.sqrt();
    // println!("sigsdt: {}", sigsdt);
    let lnS = S.ln();
    // println!("lnS: {}", lnS);

    // randomization
    let mut S_T: Vec<f64> = vec![lnS; m];
    S_T.par_iter_mut().for_each(|val| {
        let mut rng = rng();
        for _ in 0..n {
            let epsilon: f64 = rng.sample(StandardNormal);
            *val += nudt + sigsdt * epsilon;
        }
    });

    let S_T = S_T.par_iter().map(|val| val.exp()).collect::<Vec<f64>>();
    //println!("S_T: {:?}", S_T);

    // calculate the option price
    let C_T = S_T
        .par_iter()
        .map(|S_T| {
            // let S_T = val.exp();
            match option_type {
                OptionType::Call => (S_T - K).max(0.0),
                OptionType::Put => (K - S_T).max(0.0),
            }
        })
        .collect::<Vec<f64>>();

    let sum_CT = C_T.iter().sum::<f64>();
    let sum_CT2 = C_T.iter().map(|x| x.powi(2)).sum::<f64>();
    let C0 = sum_CT / m as f64 * (-r * T).exp();
    let SD =
        (((sum_CT2 - sum_CT.powi(2) / m as f64) * (-2.0 * r * T).exp()) / (m as f64 - 1.0)).sqrt();
    let SE = SD / (m as f64).sqrt();

    Ok(())
}
