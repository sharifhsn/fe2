use std::path::PathBuf;

use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum OptionType {
    Call,
    Put,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum OptionStyle {
    European,
    American,
}

#[derive(Clone)]
pub struct FiniteDifference {
    option_type: OptionType,
    option_style: OptionStyle,
    K: f64,
    T: f64,
    S: f64,
    sig: f64,
    r: f64,
    div: f64,
    epsilon: f64,
}

impl FiniteDifference {
    pub fn explicit(&self) -> f64 {
        // precompute constants
        let dt = self.epsilon / (1.0 + 3.0 * self.sig.powi(2));
        let n = (self.T / dt).ceil() as usize;
        let N = n;
        let dx = self.sig * (3.0 * dt).sqrt();
        let nu = self.r - self.div - 0.5 * self.sig.powi(2);
        let edx = dx.exp();
        let pu = 0.5 * dt * ((self.sig / dx).powi(2) + nu / dx);
        let pm = 1.0 - dt * (self.sig / dx).powi(2) - self.r * dt;
        let pd = 0.5 * dt * ((self.sig / dx).powi(2) - nu / dx);

        // initialize asset prices at maturity
        // conceptually, the mapping goes from [-N, N] to [0, 2N].rev() by j - N
        let mut St = DVector::zeros(2 * N + 1);
        St[2 * N] = self.S * (-(N as f64) * dx).exp();
        for j in (0..2 * N).rev() {
            St[j] = St[j + 1] * edx;
        }

        // initialize option values at maturity
        let mut C = DMatrix::zeros(2 * N + 1, N + 1);
        for j in 0..=2 * N {
            C[(j, N)] = 0.0f64.max(match self.option_type {
                OptionType::Call => St[j] - self.K,
                OptionType::Put => self.K - St[j],
            });
        }

        // step back through lattice
        for i in (0..n).rev() {
            // calculate discounted expectation
            for j in 1..2 * N {
                C[(j, i)] = pu * C[(j - 1, i + 1)] + pm * C[(j, i + 1)] + pd * C[(j + 1, i + 1)];
            }

            // boundary conditions, subject to option type
            C[(2 * N, i)] = C[(2 * N - 1, i)]
                + match self.option_type {
                    OptionType::Call => 0.0,
                    OptionType::Put => St[2 * N - 1] - St[2 * N],
                };
            C[(0, i)] = C[(1, i)]
                + match self.option_type {
                    OptionType::Call => St[0] - St[1],
                    OptionType::Put => 0.0,
                };

            // apply early exercise condition
            if self.option_style == OptionStyle::American {
                for j in 0..=2 * N {
                    C[(j, i)] = C[(j, i)].max(match self.option_type {
                        OptionType::Call => St[j] - self.K,
                        OptionType::Put => self.K - St[j],
                    });
                }
            }
        }
        println!("Explicit finite difference:");
        println!("n: {n}");
        println!("N: {N}");
        C[(N, 0)]
    }

    pub fn implicit(&self) -> f64 {
        // precompute constants
        let dt = self.epsilon / 2.0;
        let dx = (self.epsilon / 2.0).sqrt();
        let n = (self.T / dt).ceil() as usize;
        //let N = (3.0 * self.sig * self.T.sqrt() / dx).ceil() as usize;
        let N = n;

        let nu = self.r - self.div - 0.5 * self.sig.powi(2);
        let edx = dx.exp();
        let a = -0.5 * dt * ((self.sig / dx).powi(2) + nu / dx);
        let b = 1.0 + dt * (self.sig / dx).powi(2) + self.r * dt;
        let c = -0.5 * dt * ((self.sig / dx).powi(2) - nu / dx);

        // initialize asset prices at maturity
        // conceptually, the mapping goes from [-N, N] to [0, 2N].rev() by j - N
        let mut St = DVector::zeros(2 * N + 1);
        St[2 * N] = self.S * (-(N as f64) * dx).exp();
        for j in (0..2 * N).rev() {
            St[j] = St[j + 1] * edx;
        }

        // initialize option values at maturity
        let mut C = DMatrix::zeros(2 * N + 1, 2);
        for j in 0..=2 * N {
            C[(j, 0)] = 0.0f64.max(match self.option_type {
                OptionType::Call => St[j] - self.K,
                OptionType::Put => self.K - St[j],
            });
        }

        // compute derivative boundary condition
        let lambda_U = (St[0] - St[1])
            * match self.option_type {
                OptionType::Call => 1.0,
                OptionType::Put => 0.0,
            };
        let lambda_L = (St[2 * N] - St[2 * N - 1])
            * match self.option_type {
                OptionType::Call => 0.0,
                OptionType::Put => -1.0,
            };

        // step back through lattice
        for _ in (0..n).rev() {
            // boundary conditions, subject to option type
            Self::solve_tridiagonal_recurrence(&mut C, a, b, c, lambda_L, lambda_U);

            // update option values for the current time step
            for j in 0..=2 * N {
                C[(j, 0)] = C[(j, 1)];
            }
            //println!("{}", C);
            // apply early exercise condition
            if self.option_style == OptionStyle::American {
                for j in 0..=2 * N {
                    C[(j, 0)] = C[(j, 0)].max(match self.option_type {
                        OptionType::Call => St[j] - self.K,
                        OptionType::Put => self.K - St[j],
                    });
                }
            }
        }
        println!("Implicit finite difference:");
        println!("n: {n}");
        println!("N: {N}");
        C[(N, 0)]
    }

    pub fn crank_nicholson(&self) -> f64 {
        // precompute constants
        let dt = 2.0 * (self.epsilon / 2.0).sqrt();
        let dx = (self.epsilon / 2.0).sqrt();
        let N = (3.0 * self.sig * self.T.sqrt() / dx).ceil() as usize;
        let n = (self.T / dt).ceil() as usize;

        let nu = self.r - self.div - 0.5 * self.sig.powi(2);
        let edx = dx.exp();
        let a = -0.25 * dt * ((self.sig / dx).powi(2) + nu / dx);
        let b = 1.0 + 0.5 * dt * (self.sig / dx).powi(2) + 0.5 * self.r * dt;
        let c = -0.25 * dt * ((self.sig / dx).powi(2) - nu / dx);

        // initialize asset prices at maturity
        // conceptually, the mapping goes from [-N, N] to [0, 2N].rev() by j - N
        let mut St = DVector::zeros(2 * N + 1);
        St[2 * N] = self.S * (-(N as f64) * dx).exp();
        for j in (0..2 * N).rev() {
            St[j] = St[j + 1] * edx;
        }

        // initialize option values at maturity
        let mut C = DMatrix::zeros(2 * N + 1, 2);
        for j in 0..=2 * N {
            C[(j, 0)] = 0.0f64.max(match self.option_type {
                OptionType::Call => St[j] - self.K,
                OptionType::Put => self.K - St[j],
            });
        }

        // compute derivative boundary condition
        let lambda_U = (St[0] - St[1])
            * match self.option_type {
                OptionType::Call => 1.0,
                OptionType::Put => 0.0,
            };
        let lambda_L = (St[2 * N] - St[2 * N - 1])
            * match self.option_type {
                OptionType::Call => 0.0,
                OptionType::Put => -1.0,
            };

        // step back through lattice
        for _ in (0..n).rev() {
            // boundary conditions, subject to option type
            Self::solve_tridiagonal_recurrence(&mut C, a, b, c, lambda_L, lambda_U);

            // update option values for the current time step
            for j in 0..=2 * N {
                C[(j, 0)] = C[(j, 1)];
            }
            //println!("{}", C);
            // apply early exercise condition
            if self.option_style == OptionStyle::American {
                for j in 0..=2 * N {
                    C[(j, 0)] = C[(j, 0)].max(match self.option_type {
                        OptionType::Call => St[j] - self.K,
                        OptionType::Put => self.K - St[j],
                    });
                }
            }
        }
        C[(N, 0)]
    }
    fn solve_tridiagonal_recurrence(
        C: &mut DMatrix<f64>,
        a: f64,        // Subdiagonal coefficient (A)
        b: f64,        // Main diagonal coefficient (B)
        c: f64,        // Superdiagonal coefficient (C)
        lambda_L: f64, // Lower boundary condition
        lambda_U: f64, // Upper boundary condition
    ) {
        let mut y = C.column_mut(0);
        let n = y.len();

        // Boundary conditions
        y[0] = lambda_U;
        y[n - 1] = lambda_L;

        // Vectors to store recurrence coefficients
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];

        d[0] = y[0];
        e[0] = 1.0;

        // Forward recurrence to compute D_i and E_i
        for i in 1..n - 1 {
            let denominator = a * e[i - 1] + b;
            d[i] = (y[i] - a * d[i - 1]) / denominator;
            e[i] = -c / denominator;
        }

        // Back substitution
        let mut x = DVector::zeros(n);
        x[n - 1] = (y[n - 1] - d[n - 2]) / (e[n - 2] - 1.0);

        for i in (1..n - 1).rev() {
            x[i] = d[i] + e[i] * x[i + 1];
        }

        x[0] = y[0] + x[1];

        C.column_mut(1).copy_from(&x);
    }
    fn solve_tridiagonal_recurrence_cn(
        C: &mut DMatrix<f64>,
        a: f64,        // Subdiagonal coefficient (A)
        b: f64,        // Main diagonal coefficient (B)
        c: f64,        // Superdiagonal coefficient (C)
        lambda_L: f64, // Lower boundary condition
        lambda_U: f64, // Upper boundary condition
    ) {
        let n = C.nrows();
        let mut A = DMatrix::zeros(n, n);

        // Fill the tridiagonal matrix
        for i in 1..n - 1 {
            A[(i, i - 1)] = a;
            A[(i, i)] = b;
            A[(i, i + 1)] = c;
        }

        // Boundary conditions
        A[(0, 0)] = 1.0;
        A[(n - 1, n - 1)] = 1.0;
        let mut y = C.column_mut(0);

        // Boundary conditions
        y[0] = lambda_U;
        y[n - 1] = lambda_L;

        // Vectors to store recurrence coefficients
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];

        d[0] = y[0];
        e[0] = 1.0;

        // Forward recurrence to compute D_i and E_i
        for i in 1..n - 1 {
            let denominator = a * e[i - 1] + b;
            d[i] = (y[i] - a * d[i - 1]) / denominator;
            e[i] = -c / denominator;
        }

        // Back substitution
        let mut x = DVector::zeros(n);
        x[n - 1] = (y[n - 1] - d[n - 2]) / (e[n - 2] - 1.0);

        for i in (1..n - 1).rev() {
            x[i] = d[i] + e[i] * x[i + 1];
        }

        x[0] = y[0] + x[1];

        C.column_mut(1).copy_from(&x);
    }

    pub fn delta(&self) -> f64 {
        let epsilon = 1e-4;
        let v1 = self.clone();
        let v2 = self.clone();
        let v1 = v1.explicit();
        let v2 = v2.explicit();
        (v2 - v1) / epsilon
    }
}

fn black_scholes(
    option_type: OptionType,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    sig: f64,
    div: f64,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let d1 = (S / K).ln() + (r - div + 0.5 * sig.powi(2)) * T;
    let d1 = d1 / (sig * T.sqrt());
    let d2 = d1 - sig * T.sqrt();

    match option_type {
        OptionType::Call => {
            S * (-div * T).exp() * normal.cdf(d1) - K * (-r * T).exp() * normal.cdf(d2)
        }
        OptionType::Put => {
            K * (-r * T).exp() * normal.cdf(-d2) - S * (-div * T).exp() * normal.cdf(-d1)
        }
    }
}

pub fn a() -> PolarsResult<()> {
    const EPSILON: f64 = 1e-4;

    let fd = FiniteDifference {
        option_type: OptionType::Call,
        option_style: OptionStyle::European,
        K: 100.0,
        T: 1.0,
        S: 100.0,
        sig: 0.2,
        r: 0.06,
        div: 0.02,
        epsilon: EPSILON,
    };
    let v = fd.implicit();
    let bs_value = black_scholes(fd.option_type, fd.S, fd.K, fd.T, fd.r, fd.sig, fd.div);
    println!("Finite Difference Value: {v}");
    println!("Black-Scholes Value: {bs_value}");

    println!("Hello, world!");
    Ok(())
}
