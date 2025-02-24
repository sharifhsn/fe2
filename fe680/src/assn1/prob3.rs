#![allow(clippy::approx_constant)]
use std::io::Cursor;

use argmin::{
    core::{CostFunction, Executor},
    solver::neldermead::NelderMead,
};

use image::{codecs::png::PngDecoder, load_from_memory};
use itertools::izip;
use plotters::prelude::*;
use png::Encoder;
use polars::prelude::*;

fn data_load() -> PolarsResult<DataFrame> {
    df!(
        "time_to_next_payment" => [0.4356, 0.2644, 0.2658, 0.4342, 0.0192, 0.4753, 0.3534, 0.1000, 0.2685, 0.4342,
                                   0.2274, 0.1027, 0.2712, 0.4370, 0.4822, 0.2260, 0.4822, 0.2260, 0.2301, 0.4808,
                                   0.4932, 0.4959, 0.2397, 0.4959, 0.2397, 0.4959, 0.2397, 0.2438, 0.4945],
        "payment_frequency" => [0.5; 29],
        "time_to_maturity" => [0.4356, 0.7644, 1.2658, 1.9342, 2.0192, 2.9753, 3.3534, 3.6000, 4.2685, 4.9342,
                               5.2274, 5.6027, 6.2712, 6.9370, 7.4822, 7.7260, 8.4822, 8.7260, 9.2301, 9.9808,
                               25.4932, 26.4959, 26.7397, 27.4959, 27.7397, 28.4959, 28.7397, 29.2438, 29.9945],
        "coupon_rate" => [0.0078, 0.0078, 0.0065, 0.0053, 0.0028, 0.0065, 0.0140, 0.0165, 0.0203, 0.0165,
                          0.0440, 0.0228, 0.0265, 0.0228, 0.0340, 0.0378, 0.0265, 0.0303, 0.0328, 0.0253,
                          0.0440, 0.0465, 0.0490, 0.0428, 0.0440, 0.0340, 0.0415, 0.0428, 0.0378],
        "clean_price" => [100.30, 100.48, 100.50, 100.31, 99.78, 100.16, 102.34, 103.08, 104.19, 102.06,
                          115.91, 104.36, 105.86, 102.97, 110.53, 113.09, 103.98, 106.50, 108.00, 101.19,
                          117.58, 122.28, 126.97, 115.19, 117.47, 98.98, 112.44, 114.67, 105.75]
    )
}
const INIT_BETA_0: f64 = 0.03;
const INIT_BETA_1: f64 = -0.02;
const INIT_BETA_2: f64 = 0.02;
const INIT_LAMBDA: f64 = 1.0;

#[derive(Clone, Default)]
struct NelsonSiegel {
    df: DataFrame,
}

fn r(beta_0: f64, beta_1: f64, beta_2: f64, lambda: f64, t: f64) -> f64 {
    beta_0
        + beta_1 * ((1.0 - (-lambda * t).exp()) / (lambda * t))
        + beta_2 * ((1.0 - (-lambda * t).exp()) / (lambda * t) - (-lambda * t).exp())
}
impl CostFunction for NelsonSiegel {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        let beta_0 = param[0];
        let beta_1 = param[1];
        let beta_2 = param[2];
        let lambda = param[3];

        let lf = self.df.clone().lazy().with_column(
            map_multiple(
                move |cols| match cols {
                    [nrp, b, c, d, e] => {
                        let (nrp, b, c, d, e) =
                            (nrp.clone(), b.clone(), c.clone(), d.clone(), e.clone());

                        let num_remaining_payments = nrp.i32()?;
                        let time_to_next_payment = b.f64()?;
                        let time_to_maturity = c.f64()?;
                        let coupon_payment = d.f64()?;
                        let payment_frequency = e.f64()?;

                        let res: Float64Chunked = izip!(
                            num_remaining_payments,
                            time_to_next_payment,
                            time_to_maturity,
                            coupon_payment,
                            payment_frequency
                        )
                        .map(|(m, t1, tm, c, dt)| match (m, t1, tm, c, dt) {
                            (Some(m), Some(t1), Some(tm), Some(c), Some(dt)) => Some(
                                (1..=m)
                                        .map(|i| {
                                            let t = t1 + (i as f64 - 1.0) * dt;
                                            (-r(beta_0, beta_1, beta_2, lambda, t) * t).exp() * c
                                        })
                                        .sum::<f64>()
                                        // add principal
                                        + (-r(beta_0, beta_1, beta_2, lambda, tm) * tm).exp()
                                            * P,
                            ),
                            _ => None,
                        })
                        .collect();

                        Ok(Some(res.into_column()))
                    }
                    _ => Err(PolarsError::ComputeError(
                        "Expected exactly 5 columns".into(),
                    )),
                },
                &[
                    col("num_remaining_payments"),
                    col("time_to_next_payment"),
                    col("time_to_maturity"),
                    col("coupon_payment"),
                    col("payment_frequency"),
                ],
                GetOutput::from_type(DataType::Float64),
            )
            .alias("theoretical_price"),
        );
        let cost_expr = (col("dirty_price") - col("theoretical_price")).pow(2.0) * col("weight");

        let cost_df = lf.select([cost_expr.sum().alias("cost")]).collect()?;

        let cost: f64 = cost_df.column("cost")?.f64()?.get(0).unwrap();
        Ok(cost)
    }
}

const P: f64 = 100.0; // principal

pub fn a() -> PolarsResult<()> {
    let df = data_load()?;
    println!("{df}");

    let lf = df
        .lazy()
        .with_column(
            (col("coupon_rate") * col("payment_frequency") * lit(P)).alias("coupon_payment"),
        )
        .with_column(
            (col("clean_price")
                + col("coupon_payment") * (col("payment_frequency") - col("time_to_next_payment"))
                    / col("payment_frequency"))
            .alias("dirty_price"),
        )
        .with_column(
            ((col("time_to_maturity") / col("payment_frequency")).floor() + lit(1))
                .cast(DataType::Int32)
                .alias("num_remaining_payments"),
        )
        .with_column(lit(1.0).alias("weight"));
    let nelson_siegel = NelsonSiegel {
        df: lf.collect().unwrap(),
    };
    let initial_params = vec![INIT_BETA_0, INIT_BETA_1, INIT_BETA_2, INIT_LAMBDA];
    // Construct initial simplex for Nelder-Mead
    let perturbation = 0.1;
    let mut simplex = vec![initial_params.clone()];

    for i in 0..initial_params.len() {
        let mut new_vertex = initial_params.clone();
        new_vertex[i] += perturbation;
        simplex.push(new_vertex);
    }
    // Create Nelder-Mead solver with valid parameter type
    let solver: NelderMead<Vec<f64>, f64> =
        NelderMead::new(simplex).with_sd_tolerance(1e-10).unwrap();
    let res = Executor::new(nelson_siegel, solver).run().unwrap();
    let state = res.state().param.clone().unwrap();
    let beta_0 = state[0];
    let beta_1 = state[1];
    let beta_2 = state[2];
    let lambda = state[3];
    println!("{beta_0}, {beta_1}, {beta_2}, {lambda}");
    let t_values: Vec<f64> = (5..=300).map(|x| x as f64 / 10.0).collect();
    let yield_values: Vec<f64> = t_values
        .iter()
        .map(|&t| r(beta_0, beta_1, beta_2, lambda, t))
        .collect();

    // Set up the plot
    let root = BitMapBackend::new("yield_curve_ns.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Yield Curve (Nelson-Siegel)", ("sans-serif", 25))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..30.0, -0.0..0.05)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Plot the yield curve
    chart
        .draw_series(LineSeries::new(
            t_values
                .iter()
                .zip(yield_values.iter())
                .map(|(&t, &y)| (t, y)),
            &BLUE,
        ))
        .unwrap()
        .label("Yield")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    // Configure the legend
    chart
        .configure_series_labels()
        .background_style(WHITE)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Yield curve saved as 'yield_curve_ns.png'");

    Ok(())
}
