use std::fs::File;
use std::io::Write;

use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::LBFGS;
use argmin_math::Error;
use nalgebra::{dmatrix, DMatrix, DVector};
use plotters::evcxr::evcxr_figure;
use plotters::prelude::*;

// Define the objective function with penalty
struct F1WithPenalty;

impl CostFunction for F1WithPenalty {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        let x1 = x[0];
        let x2 = x[1];
        let x3 = x[2];

        let objective = x1 + x2 + 2.0 * x3.powi(2);
        let constraint_penalty = (x1 - 1.0).powi(2) + (x1.powi(2) + x2.powi(2) - 1.0).powi(2);

        Ok(objective + 1e12 * constraint_penalty)
    }
}

// Separate the original f1(x) (without penalty)
fn f1(x: &Vec<f64>) -> f64 {
    x[0] + x[1] + 2.0 * x[2].powi(2)
}

pub fn a() -> Result<(), Box<dyn std::error::Error>> {
    // Initial guess: [1, 0, 0]
    let init_param = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    // Set up the solver
    let solver = NelderMead::new(init_param);

    // Run the solver
    let res = Executor::new(F1WithPenalty, solver).run()?;

    let best_param = res.state().param.clone().unwrap();

    println!("Best solution found: {:?}", &best_param);
    println!("Value of f1 at solution: {}", f1(&best_param));

    Ok(())
}

struct F2 {
    rhs: f64,
}

impl CostFunction for F2 {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        let x1 = x[0];
        let x2 = x[1];
        let objective = 2.0 * x1.powi(2) + x2.powi(2);
        let penalty = 1e8 * (x1 + x2 - self.rhs).powi(2);
        Ok(objective + penalty)
    }
}

impl Gradient for F2 {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Error> {
        let x1 = x[0];
        let x2 = x[1];

        // Gradient of objective
        let grad_obj_x1 = 4.0 * x1;
        let grad_obj_x2 = 2.0 * x2;

        // Gradient of penalty
        let common_penalty_grad = 2.0 * 1e2 * (x1 + x2 - self.rhs);

        Ok(vec![
            grad_obj_x1 + common_penalty_grad,
            grad_obj_x2 + common_penalty_grad,
        ])
    }
}

fn lagrangian(a: DMatrix<f64>, rhs: DVector<f64>) -> DVector<f64> {
    a.lu()
        .solve(&rhs)
        .expect("System should have unique solution")
}

fn f2(x: &Vec<f64>) -> f64 {
    2.0 * x[0].powi(2) + x[1].powi(2)
}

pub fn b() -> Result<(), Box<dyn std::error::Error>> {
    let data = dmatrix![
        4.0, 0.0, 1.0;
        0.0, 2.0, 1.0;
        1.0, 1.0, 0.0;
    ];
    // === Lagrangian solution ===
    let rhs1 = DVector::from_vec(vec![0.0, 0.0, 1.0]);
    let rhs1_05 = DVector::from_vec(vec![0.0, 0.0, 1.05]);

    let solution1 = lagrangian(data.clone(), rhs1);
    let solution1_05 = lagrangian(data.clone(), rhs1_05);

    let x1_rhs1 = solution1[0];
    let x2_rhs1 = solution1[1];
    let x1_rhs1_05 = solution1_05[0];
    let x2_rhs1_05 = solution1_05[1];

    println!(
        "Lagrangian solution (rhs = 1): x1 = {:.4}, x2 = {:.4}",
        x1_rhs1, x2_rhs1
    );
    println!(
        "Lagrangian solution (rhs = 1.05): x1 = {:.4}, x2 = {:.4}",
        x1_rhs1_05, x2_rhs1_05
    );

    // === Gradient-based numerical optimization ===
    let init_param1 = vec![1.0 / 3.0, 2.0 / 3.0];
    let init_param2 = vec![0.35, 0.70];

    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.2)?);

    let solver1 = LBFGS::new(linesearch.clone(), 5);
    let solver2 = LBFGS::new(linesearch, 5);
    let res1 = Executor::new(F2 { rhs: 1.0 }, solver1)
        .configure(|state| state.param(init_param1))
        .run()?;

    let res2 = Executor::new(F2 { rhs: 1.05 }, solver2)
        .configure(|state| state.param(init_param2))
        .run()?;

    let best1 = res1.state().param.clone().unwrap();
    let best2 = res2.state().param.clone().unwrap();

    println!(
        "\nGradient solution (rhs = 1): x1 = {:.4}, x2 = {:.4}",
        best1[0], best1[1]
    );
    println!(
        "Gradient solution (rhs = 1.05): x1 = {:.4}, x2 = {:.4}",
        best2[0], best2[1]
    );

    println!("\nf* (rhs = 1): {:.4}", f2(&best1));
    println!("f* (rhs = 1.05): {:.4}", f2(&best2));
    println!("Difference: {:.6}", f2(&best2) - f2(&best1));
    Ok(())
}

fn R_p(
    sigma1: f64,
    sigma2: f64,
    rho12: f64,
    sigma_t: f64,
    rho1: f64,
    rho2: f64,
) -> Result<(f64, f64, f64), String> {
    let a = sigma1.powi(2) + sigma2.powi(2) - 2.0 * rho12 * sigma1 * sigma2;
    let b = sigma2.powi(2) - rho12 * sigma1 * sigma2;
    let c = sigma2.powi(2) - sigma_t.powi(2);

    let discriminant = b.powi(2) - a * c;
    if discriminant < 0.0 {
        return Err("No real solution exists for the given inputs.".to_owned());
    }

    let sqrt_discriminant = discriminant.sqrt();

    // Calculate the two possible solutions for ω1
    let omega1_1 = (b + sqrt_discriminant) / a;
    let omega1_2 = (b - sqrt_discriminant) / a;

    // Choose the solution that satisfies the constraints (e.g., 0 <= ω1 <= 1)
    let omega1 = if (0.0..=1.0).contains(&omega1_1) {
        omega1_1
    } else if (0.0..=1.0).contains(&omega1_2) {
        omega1_2
    } else {
        return Err("No valid solution for ω1 in the range [0, 1].".to_owned());
    };

    let omega2 = 1.0 - omega1;

    // Calculate the optimal return
    let rp = rho1 * omega1 + rho2 * omega2;

    Ok((omega1, omega2, rp))
}

pub fn c() -> Result<(), Box<dyn std::error::Error>> {
    let rho1 = 0.05;
    let rho2 = 0.10;
    let sigma1 = 0.10;
    let sigma2 = 0.20;
    let rho12 = -0.5;

    let sigma_t_values: Vec<f64> = (4..=60).map(|x| x as f64 / 200.0).collect();
    let mut rp_values = Vec::new();

    for sigma_t in &sigma_t_values {
        match R_p(sigma1, sigma2, rho12, *sigma_t, rho1, rho2) {
            Ok((_, _, rp)) => {
                rp_values.push(rp);
            }
            Err(e) => {
                //println!("Error: {}", e);
                rp_values.push(0.0); // Push a default value in case of error
                continue;
            }
        };
    }

    let root = evcxr_figure((800, 600), |root| {
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Efficient Frontier", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..0.30, 0.0..0.12)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                sigma_t_values.into_iter().zip(rp_values),
                &RED,
            ))
            .unwrap()
            .label("Efficient Frontier")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));
        for i in 0..3 {
            chart
                .draw_series(PointSeries::of_element(
                    vec![((iu.cov[(i, i)]).sqrt(), iu.rho[i])],
                    5,
                    &RED,
                    &|c, s, st| {
                        return EmptyElement::at(c)
                            + Circle::new((0, 0), s, st.filled())
                            + Text::new(format!("S{i}"), (10, 0), ("sans-serif", 15));
                    },
                ))
                .unwrap();
        }
        chart
            .configure_series_labels()
            .border_style(BLACK)
            .draw()
            .unwrap();

        Ok(())
    });

    root.evcxr_display();
    Ok(())
}
