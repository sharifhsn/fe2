use core::time;
use std::arch::x86_64;

use polars::prelude::*;



// pub fn data_load() -> PolarsResult<DataFrame> {
//     df!(
//         "time_to_maturity" => [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0],
//         "yield_rate" => [0.0150, 0.0160, 0.0180, 0.0210, 0.0240, 0.0330, 0.03740, 0.0405, 0.0435]
//     )
// }

// pub fn compute_splines() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
//     let n = X.len();
//     let mut h = vec![0.0; n - 1];
//     let mut alpha = vec![0.0; n - 1];
//     let mut l = vec![1.0; n];
//     let mut mu = vec![0.0; n];
//     let mut z = vec![0.0; n];

//     // Step 1: Calculate h and alpha
//     for i in 0..n - 1 {
//         h[i] = X[i + 1] - X[i];
//         if h[i] == 0.0 {
//             panic!("x values must be distinct");
//         }
//         alpha[i] = (3.0 / h[i]) * (Y[i + 1] - Y[i]) - (3.0 / h[i - 1]) * (Y[i] - Y[i - 1]);
//     }

//     // Step 2: Decompose the matrix
//     for i in 1..n - 1 {
//         l[i] = 2.0 * (X[i + 1] - X[i - 1]) - h[i - 1] * mu[i - 1];
//         mu[i] = h[i] / l[i];
//         z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
//     }

//     // Step 3: Back substitution
//     let mut c = vec![0.0; n];
//     let mut b = vec![0.0; n - 1];
//     let mut d = vec![0.0; n - 1];

//     l[n - 1] = 1.0;
//     z[n - 1] = 0.0;
//     c[n - 1] = 0.0;

//     for j in (0..n - 2).rev() {
//         c[j] = z[j] - mu[j] * c[j + 1];
//         b[j] = (Y[j + 1] - Y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
//         d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
//     }

//     // Combine b, c, and d into a single vector
//     (b, c, d, Y.to_vec())
// }

// fn interpolate_yield_curve(x: f64, x_vals: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> f64 {
//     let n = x_vals.len();
//     for i in 0..n - 1 {
//         if x >= x_vals[i] && x <= x_vals[i + 1] {
//             let dx = x - x_vals[i];
//             return b[i] + c[i] * dx + d[i] * dx.powi(2);
//         }
//     }
//     panic!("x is out of bounds");
// }

use cubic_spline::{Points, Point, SplineOpts, TryFrom};
pub fn a() {
    let time_to_maturity = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0];
    let yield_rate = vec![0.0150, 0.0160, 0.0180, 0.0210, 0.0240, 0.0330, 0.03740, 0.0405, 0.0435];

    let src: Vec<(f64, f64)> = time_to_maturity.into_iter().zip(yield_rate).collect();
    let opts = SplineOpts::new();

    let mut points = <Points as cubic_spline::TryFrom>::try_from(&src).unwrap();


    let (b, c, d, a) = compute_splines();

    let interpolation_points = vec![0.75, 1.5, 2.5, 4.0, 6.0, 8.0, 15.0];
    for &point in &interpolation_points {
        let interpolated_yield = interpolate_yield_curve(point, &X, &b, &c, &d);
        println!("Time to Maturity: {:.2}, Interpolated Yield Rate: {:.4}", point, interpolated_yield);
    }
}
