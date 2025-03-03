use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::E;

enum OptionType {
    Call,
    Put,
}

/// European Up-and-Out Call Barrier Option using an Additive Trinomial Tree
#[derive(Debug)]
struct AdditiveTrinomialTreeBarrierOption {
    s0: f64,    // Initial stock price
    k: f64,     // Strike price
    h: f64,     // Barrier level
    t: f64,     // Time to maturity (years)
    r: f64,     // Risk-free rate
    sigma: f64, // Volatility
    n: usize,   // Number of steps in the tree
}

impl AdditiveTrinomialTreeBarrierOption {
    fn price(&self) -> f64 {
        let dt = self.t / self.n as f64;
        let dx = self.sigma * (3.0 * dt).sqrt(); // Log price step size

        let drift = self.r - 0.5 * self.sigma.powi(2);
        let pu = ((self.sigma.powi(2) * dt + drift.powi(2) * dt.powi(2)) / dx.powi(2)
            + drift * dt / dx)
            / 2.0;
        let pm = 1.0 - (self.sigma.powi(2) * dt + drift.powi(2) * dt.powi(2)) / dx.powi(2);
        let pd = ((self.sigma.powi(2) * dt + drift.powi(2) * dt.powi(2)) / dx.powi(2)
            - drift * dt / dx)
            / 2.0;
        let discount = (-self.r * dt).exp();

        let x0 = self.s0.ln(); // Initial log price

        // Initialize log price tree
        let mut log_tree = DMatrix::zeros(2 * self.n + 1, self.n + 1);
        for j in 0..=self.n {
            for i in 0..=2 * j {
                log_tree[(i, j)] = x0 + (i as f64 - j as f64) * dx;
            }
        }
        // Convert log prices back to stock prices
        let stock_tree = log_tree.map(|x| E.powf(x));
        // println!("{}", stock_tree.clone());
        // Initialize option values at maturity
        let mut option_tree = DMatrix::zeros(2 * self.n + 1, self.n + 1);
        for i in 0..=2 * self.n {
            let s = stock_tree[(i, self.n)];
            option_tree[(i, self.n)] = if s < self.h {
                f64::max(s - self.k, 0.0) // / (discount.powi(self.n as i32))
            } else {
                0.0
            };
        }
        // println!("{}", option_tree);
        // Backward induction
        for j in (0..self.n).rev() {
            for i in 0..=2 * j {
                let s = stock_tree[(i, j)];
                if s >= self.h {
                    option_tree[(i, j)] = 0.0;
                } else {
                    option_tree[(i, j)] = discount
                        * (pu * option_tree[(i, j + 1)]
                            + pm * option_tree[(i + 1, j + 1)]
                            + pd * option_tree[(i + 2, j + 1)]);
                }
            }
        }
        // println!("{}", option_tree);
        option_tree[(0, 0)]
    }
}

fn black_scholes(option_type: OptionType, s0: f64, σ: f64, τ: f64, k: f64, r: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let d1 = (s0.ln() - k.ln() + (r + 0.5 * σ.powi(2)) * τ) / (σ * τ.sqrt());
    let d2 = d1 - σ * τ.sqrt();

    match option_type {
        OptionType::Call => s0 * normal.cdf(d1) - k * E.powf(-r * τ) * normal.cdf(d2),
        OptionType::Put => k * E.powf(-r * τ) * normal.cdf(-d2) - s0 * normal.cdf(-d1),
    }
}

fn up_and_out(s0: f64, σ: f64, τ: f64, k: f64, r: f64, h: f64, delta: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let v = r - delta - σ.powi(2) / 2.0;
    let dbs = |s0: f64, k: f64| ((s0 / k).log(E) + v * τ) / (σ * τ.sqrt());
    black_scholes(OptionType::Call, s0, σ, τ, k, r)
        - black_scholes(OptionType::Call, s0, σ, τ, h, r)
        - (h - k) * ((-r * τ).exp()) * normal.cdf(dbs(s0, h))
        - (h / s0).powf(2.0 * v / σ.powi(2))
            * (black_scholes(OptionType::Call, h.powi(2) / s0, σ, τ, k, r)
                - black_scholes(OptionType::Call, h.powi(2) / s0, σ, τ, h, r)
                - (h - k) * ((-r * τ).exp()) * normal.cdf(dbs(h, s0)))
}

pub fn a() {
    let s0 = 10.0;
    let k = 10.0;
    let t = 0.3;
    let sigma = 0.2;
    let r = 0.01;
    let h = 11.0;
    let delta = 0.0;
    let n = 2000; // Number of time steps

    let opt = AdditiveTrinomialTreeBarrierOption {
        s0,
        k,
        h,
        t,
        r,
        sigma,
        n,
    };
    println!("{}", opt.price());

    println!("Theoretical {}", up_and_out(s0, sigma, t, k, r, h, delta));
}
