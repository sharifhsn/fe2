use statrs::distribution::{Normal, ContinuousCDF};
use rand::prelude::*;
use rand_distr::{Normal as RandNormal, Distribution};
use nalgebra::DMatrix;
use plotters::prelude::*;

use std::f64::consts::PI;
use std::f64::EPSILON;

use super::biv::BvNormal;

fn genz_phi_2(b1: f64, b2: f64, rho: f64) -> f64 {
    let h = -b1;
    let k = -b2;
    todo!()
}
pub struct GaussianLatentVariableMultiName<const Nc: usize> {
    pub R: [f64; Nc], // recovery rate
    pub beta: [f64; Nc], // market exposure
    pub p: [f64; Nc], // unconditional probability of default
}

impl<const N: usize> GaussianLatentVariableMultiName<N> {
    pub fn new(R: [f64; N], beta: [f64; N], p: [f64; N]) -> Self {
        GaussianLatentVariableMultiName { R, beta, p }
    }

    pub fn simulate(&self, n: usize) -> Vec<[f64; N]> {
        let c = DMatrix::from_fn(N, N, |i, j| {
            self.beta[i] * self.beta[j]
        });
        let normal = Normal::new(0.0, 1.0).unwrap();

        todo!()
    }
}

pub fn a() {
    let x = 0.3;
    // pbivnorm(prob, lower, upper_a, upper_b, infin, correl);
}
// ————————————————
// 3) Routine to solve for y given x, target p and ρ via bisection
// ————————————————
fn find_y_for_level(x: f64, rho: f64, target: f64) -> f64 {
    let bv = BvNormal::new();
    let mut lo = -5.0;
    let mut hi =  5.0;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let p   = bv.cdf(x, mid, rho);
        if p < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ————————————————
// 4) Main: heatmap + 0.5‐contour for ρ = −0.7
// ————————————————
pub fn test_biv() -> Result<(), Box<dyn std::error::Error>> {
    let width = 600;
    let height = 600;
    let rho   = -0.7_f64;
    let target_level = 0.5_f64;
    let bv = BvNormal::new();

    // 4a) Set up the drawing area (PNG file)
    let root = BitMapBackend::new("bv2_heatmap.png", (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    // 4b) Chart with axes
    let drawing = root.margin(40, 40, 40, 40);
    let mut chart = ChartBuilder::on(&drawing)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(-3.0..3.0, -3.0..3.0)?;
    chart.configure_mesh().draw()?;

    // 4c) Heatmap: draw each pixel by raw px,py
    for px in 0..width {
        for py in 0..height {
            // invert Y for image coords
            let py_img = height - 1 - py;
            // map to world coords
            let x = -3.0 + 6.0 * (px as f64) / ((width - 1) as f64);
            let y = -3.0 + 6.0 * (py_img as f64) / ((height - 1) as f64);
            let p = bv.cdf(x, y, rho);
            let shade = (255.0 * (1.0 - p)).clamp(0.0, 255.0) as u8;
            root.draw_pixel(((width - px) as i32, py_img as i32), &RGBColor(shade, shade, shade))?;
        }
    }

    // 4d) Overlay the 50% contour line
    let mut contour_points: Vec<(u32, u32)> = Vec::with_capacity(width as usize);
    for i in 0..width {
        let x = -3.0 + 6.0 * (i as f64) / ((width - 1) as f64);
        let px = ((x + 3.0) * ((width - 1) as f64) / 6.0).round();
        let y = find_y_for_level(x, rho, target_level);
        let py = ((y + 3.0) * ((height - 1) as f64) / 6.0).round();
        contour_points.push((width - px, py));
    }

    chart
        .draw_series(LineSeries::new(contour_points, &RED.mix(0.8)))?
        .label("p = 0.5")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    // 4e) Legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Wrote bv2_heatmap.png (ρ = {:.2})", rho);

    Ok(())
}
