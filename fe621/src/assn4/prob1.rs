use nalgebra::DMatrix;
use polars::prelude::*;
use rand::{prelude::*, rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
enum OptionType {
    Call,
    Put,
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
    let nudt = (r - div - 0.5 * sig.powi(2)) * dt;
    let sigsdt = sig * dt.sqrt();
    let lnS = S.ln();

    // randomization
    let mut S_T: Vec<f64> = vec![lnS; m];
    S_T.par_iter_mut().for_each(|val| {
        let mut rng = rng();
        for _ in 0..n {
            let epsilon: f64 = rng.sample(StandardNormal);
            *val += nudt + sigsdt * epsilon;
        }
    });

    // calculate the option price
    let C_T = S_T
        .par_iter()
        .map(|val| {
            let S_T = val.exp();
            match option_type {
                OptionType::Call => (S_T - K).max(0.0),
                OptionType::Put => (K - S_T).max(0.0),
            }
        })
        .collect::<Vec<f64>>();

    let sum_CT = C_T.iter().sum::<f64>();
    let sum_CT2 = C_T.iter().map(|x| x.powi(2)).sum::<f64>();
    let C0 = (sum_CT / m as f64).exp() * (-r * T).exp();
    let SD =
        ((sum_CT2 - sum_CT.powi(2) / m as f64).sqrt() * (-2.0 * r * T).exp()) / (m as f64 - 1.0);
    let SE = SD / (m as f64).sqrt();

    Ok(())
}
