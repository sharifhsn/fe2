use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use rand::distr::Distribution;
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

pub struct HullWhite {
    pub a: f64,
    pub sigma: f64,
    pub L: f64,
    pub K: f64,
    pub T: f64,
    pub s: f64,
    pub z_curve: Vec<f64>,
    pub f_curve: Vec<f64>,
}

impl HullWhite {
    pub fn new(a: f64, sigma: f64, L: f64, K: f64, T: f64, s: f64, z_curve: Vec<f64>) -> Self {
        Self {
            a,
            sigma,
            L,
            K,
            T,
            s,
            f_curve: z_to_f(&z_curve),
            z_curve,
        }
    }

    pub fn z_to_f(z: &Vec<f64>) -> Vec<f64> {
        let mut f = vec![0.0; z.len()];
        f[0] = z[0];
        for i in 1..z.len() {
            let prod: f64 = f[..i].iter().map(|x| 1.0 / (1.0 + x)).product();
            f[i] = (1.0 + z[i]).powi(i as i32 + 1) * prod - 1.0;
        }
        f
    }

    /// P(0, T) = A(0, T) * e^(-B(0, T) * r(0))
    pub fn P(&self, T: f64) -> f64 {
        self.A(0.0, T) * (-self.B(0.0, T) * self.z_curve[0]).exp()
    }

    /// ln A(t, T) = ln(P(0, T) / P(0, t)) + B(t, T) * F(t) - sigma^2 * (e^(-aT) - e^(-at))^2 * (e^(2at) - 1) / 4a^3
    fn A(&self, t: f64, T: f64) -> f64 {
        ((P(self.z_curve[T as usize], 0.0, T) / P(self.z_curve[t as usize], 0.0, t)).ln()
            + self.B(t, T) * self.f_curve[t as usize]
            - self.sigma.powi(2)
                * ((-self.a * T).exp() - (-self.a * t).exp()).powi(2)
                * ((2.0 * self.a * t).exp() - 1.0)
                / (4.0 * self.a.powi(3)))
        .exp()
    }

    /// B(t, T) = (1 - e^(-a(T - t))) / a
    fn B(&self, t: f64, T: f64) -> f64 {
        (1.0 - (-self.a * (T - t)).exp()) / self.a
    }

    pub fn bond_option_price(&self, option_type: OptionType) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let sigma_P = self.sigma / self.a
            * (1.0 - (-self.a * (self.s - self.T)).exp())
            * ((1.0 - (-2.0 * self.a * self.T).exp()) / (2.0 * self.a)).sqrt();

        let h =
            ((self.L * self.P(self.s)) / (self.P(self.T) * self.K)).ln() / sigma_P + sigma_P / 2.0;

        match option_type {
            OptionType::Call => {
                self.L * self.P(self.s) * normal.cdf(h)
                    - self.K * self.P(self.T) * normal.cdf(h - sigma_P)
            }
            OptionType::Put => {
                self.K * self.P(self.T) * normal.cdf(-h + sigma_P)
                    - self.L * self.P(self.s) * normal.cdf(-h)
            }
        }
    }
}

pub fn P(r: f64, t: f64, T: f64) -> f64 {
    (-r * (T - t)).exp()
}

pub fn z_to_f(z: &Vec<f64>) -> Vec<f64> {
    let mut f = vec![0.0; z.len()];
    f[0] = z[0];
    for i in 1..z.len() {
        let prod: f64 = f[..i].iter().map(|x| 1.0 / (1.0 + x)).product();
        f[i] = (1.0 + z[i]).powi(i as i32 + 1) * prod - 1.0;
    }
    f
}

pub fn a() -> f64 {
    let a = 0.08;
    let sigma = 0.02;
    let T = 1.0;
    let s = 5.0;
    let L = 100.0;
    let K = 70.0;
    let z_curve = vec![0.05; s as usize + 1];
    // let a = 0.1;
    // let sigma = 0.015;
    // let T = 1.0;
    // let s = 3.0;
    // let L = 100.0;
    // let K = 87.0;
    // let z_curve = vec![0.05; s as usize + 1];

    let hull_white =
        HullWhite::new(a, sigma, L, K, T, s, z_curve).bond_option_price(OptionType::Call);
    println!("Hull White: {}", hull_white);
    hull_white
}
