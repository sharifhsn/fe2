use nalgebra::DMatrix;
use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal as RandNormal};
use statrs::distribution::{ContinuousCDF, Normal};

use super::biv::BvNormal;

fn genz_phi_2(b1: f64, b2: f64, rho: f64) -> f64 {
    let h = -b1;
    let k = -b2;
    todo!()
}
pub struct GaussianLatentVariableMultiName<const Nc: usize> {
    pub R: [f64; Nc],    // recovery rate
    pub beta: [f64; Nc], // market exposure
    pub p: [f64; Nc],    // unconditional probability of default
}

impl<const N: usize> GaussianLatentVariableMultiName<N> {
    pub fn new(R: [f64; N], beta: [f64; N], p: [f64; N]) -> Self {
        GaussianLatentVariableMultiName { R, beta, p }
    }

    pub fn simulate(&self, n: usize) -> Vec<[f64; N]> {
        let c = DMatrix::from_fn(N, N, |i, j| self.beta[i] * self.beta[j]);
        let normal = Normal::new(0.0, 1.0).unwrap();

        todo!()
    }
}

pub fn a() {
    let bv = BvNormal::new();
    let result = bv.cdf(0.2, 0.5, 0.95);
    println!("CDF(0.2, 0.5, 0.95) = {}", result);
}
