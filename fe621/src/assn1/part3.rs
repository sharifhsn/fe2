use rayon::prelude::*;
use std::f64::consts::PI;

trait Quadrature {
    fn trapezoidal(&self, a: f64, n: usize) -> f64;
    fn simpsons(&self, a: f64, n: usize) -> f64;
}

impl<F> Quadrature for F
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    fn trapezoidal(&self, a: f64, n: usize) -> f64 {
        let h = (2.0 * a) / (n as f64);

        let sum: f64 = (1..n)
            .into_par_iter()
            .map(|i| {
                let x_i = -a + i as f64 * h;
                self(x_i)
            })
            .sum();

        h * (0.5 * (self(-a) + self(a)) + sum)
    }
    fn simpsons(&self, a: f64, n: usize) -> f64 {
        let h = (2.0 * a) / (n as f64);

        let sum: f64 = (1..n)
            .into_par_iter()
            .map(|i| {
                let x_i = -a + i as f64 * h;
                let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
                weight * self(x_i)
            })
            .sum();

        (h / 3.0) * (self(-a) + self(a) + sum)
    }
}

/// Compute the truncation error: I_N - π
fn compute_error(a: f64, n: usize) {
    let integral_trap = sinc.trapezoidal(a, n);
    let integral_simp = sinc.simpsons(a, n);

    let error_trap = integral_trap - PI;
    let error_simp = integral_simp - PI;

    println!(
        "a = {:e}, N = {:e} | Trapezoidal Error: {:.5e}, Simpson's Error: {:.5e}",
        a, n as f64, error_trap, error_simp
    );
}

/// Define the function sinc(x) = sin(x)/x, with proper handling of x = 0
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0 // Define sinc(0) = 1
    } else {
        x.sin() / x
    }
}

/// Find the number of steps until convergence for ε = 10^{-4}
fn convergence_steps(a: f64, epsilon: f64) {
    let mut n = 10_000;
    let mut prev_trap = sinc.trapezoidal(a, n);
    let mut prev_simp = sinc.simpsons(a, n);
    let mut iter_trap = 0;
    let mut iter_simp = 0;

    loop {
        n += 10_000; // Double N for next iteration
        let current_trap = sinc.trapezoidal(a, n);

        iter_trap += 1;

        if (current_trap - prev_trap).abs() < epsilon {
            println!(
                "Converged for a = {:e} | Trapezoidal in {} iterations",
                a, iter_trap
            );
            break;
        }

        prev_trap = current_trap;
    }
    let mut n = 10_000;
    loop {
        n += 10_000; // Double N for next iteration
        let current_simp = sinc.simpsons(a, n);

        iter_simp += 1;

        if (current_simp - prev_simp).abs() < epsilon {
            println!(
                "Converged for a = {:e} | Simpson’s in {} iterations",
                a, iter_simp
            );
            break;
        }

        prev_simp = current_simp;
    }
}

pub fn a() {
    let a = 1e6; // Large interval
    let n = 1_000_000; // Number of intervals (should be large)

    let integral_trap = sinc.trapezoidal(a, n);
    let integral_simp = sinc.simpsons(a, n);

    println!("Trapezoidal Rule Approximation: {}", integral_trap);
    println!("Simpson's Rule Approximation: {}", integral_simp);

    let a_values = [1e6, 1e7, 1e8, 1e9]; // Different integration limits
    let n_values = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]; // Different sample sizes

    for &a in &a_values {
        for &n in &n_values {
            compute_error(a, n);
        }
    }

    let a_values = [1e6, 1e7, 1e8, 1e9]; // Different integration limits
    let epsilon = 1e-4; // Convergence threshold

    for &a in &a_values {
        convergence_steps(a, epsilon);
    }
}
