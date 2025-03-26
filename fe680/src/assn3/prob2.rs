use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};
const BP: f64 = 0.0001;

enum BranchMode {
    Up,
    Middle,
    Down,
}

struct VasicekTrinomialTree {
    r0: f64,    // Initial short rate
    theta: f64, // Long-term mean
    k: f64,     // Mean reversion speed
    sigma: f64, // Volatility
    dt: f64,    // Time step
    T: f64,     // Time to maturity
    n: usize,   // Number of steps
}

impl VasicekTrinomialTree {
    fn terminal_rates(&self) -> Vec<f64> {
        let dR = self.sigma * (3.0 * self.dt).sqrt();
        let branch_mode = BranchMode::Middle;

        todo!()
    }
}

pub fn a() {
    let k = 0.025;
    let dt = 1.0 / 12.0;
    let T = 6.0 / 12.0;
    let sigma = 126.0 * BP;
    let r_0 = 0.05121;
    let theta = 0.15339;
}
