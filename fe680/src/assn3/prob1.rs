use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use rand::distr::Distribution;
use rand_distr::Normal;

pub struct TermStructureAdjustment {
    pub times: Vec<f64>,
    pub adjustments: Vec<f64>,
}

impl TermStructureAdjustment {
    pub fn new(
        times: Vec<f64>,
        adjustments: Vec<f64>,
    ) -> Self {
        assert_eq!(times.len(), adjustments.len(), "Times and adjustments must have the same length");
        Self { times, adjustments }
    }

    pub fn get_adjustment(
        &self,
        time: f64,
    ) -> Option<f64> {
        self.times
            .iter()
            .position(|&t| t == time)
            .map(|idx| self.adjustments[idx])
    }
}

impl std::ops::Index<f64>
    for TermStructureAdjustment
{
    type Output = f64;

    fn index(&self, time: f64) -> &Self::Output {
        let idx = self.times.iter().position(|&t| t == time)
            .expect("Time not found in term structure adjustments");
        &self.adjustments[idx]
    }
}

pub struct HullWhite {
    pub a: f64,
    pub sigma: f64,
    pub r: f64,
    pub L: f64,
    pub K: f64,
    pub theta: TermStructureAdjustment,
}

impl HullWhite {
    const DT: f64 = 1e-5;
    pub fn dr(&self, t: f64) -> f64 {
        let normal = Normal::new(0.0, self.sigma)
            .expect(
                "Invalid normal distribution",
            );
        (self.theta[t] - self.a * self.r)
            * Self::DT
            + normal.sample(&mut rand::rng())
                * (Self::DT).sqrt()
    }

    pub fn tree(&self) {
        let dR =
            self.sigma * (3.0 * Self::DT).sqrt();
        todo!()
    }
}

pub fn a() -> PolarsResult<DataFrame> {
    let df = df!(
        "a" => [0.08],
        "sigma" => [0.02],
        "T" => [1.0],
        "r" => [0.05],
        "L" => [100.0],
        "K" => [70.0]
    )?;

    todo!()
}
