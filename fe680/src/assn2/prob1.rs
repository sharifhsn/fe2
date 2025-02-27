#![allow(non_snake_case)]

use crate::util::N;
use itertools::izip;
use polars::prelude::*;

use std::f64::consts::E;
// 10.1252 = cash
// 12.4560 = quoted
// 11.2522
pub fn a() -> PolarsResult<DataFrame> {
    let mut df = df!(
        "name" => ["example", "problem 1"],
        "T" => [10.0/12.0, 0.5],
        "B_0" => [960.0, 904.0],
        "K" => [1000.0, 900.0],
        "sigma" => [0.09, 0.05],
        "r_m" => [0.10, 0.025],
        "payment frequency" => [0.5, 0.5],
        "coupon payment" => [50.0, 10.0],
        "time to next payment" => [0.25, 0.25],
        "option type" => ["call", "put"]
    )?;
    // Add the interest rate curve separately,
    // as the df! macro does not support lists
    df.with_column(Column::new(
        "r curve".into(),
        &[
            Series::new(
                "".into(),
                &[0.09, 0.095],
            ),
            Series::new("".into(), &[0.025]),
        ],
    ))?;

    let name = col("name");
    let T = col("T");
    let B_0 = col("B_0");
    let K = col("K");
    // let K = col("K") + (col("coupon payment"));
    let sigma = col("sigma");
    let r_m = col("r_m");
    let payment_frequency =
        col("payment frequency");
    let coupon_payment = col("coupon payment");
    let time_to_next_payment =
        col("time to next payment");
    let option_type = col("option type");
    let r_curve = col("r curve");

    let time_of_last_payment =
        time_to_next_payment.clone()
            + (((T.clone()
                - time_to_next_payment.clone())
                / payment_frequency.clone())
            .floor()
            .cast(DataType::Float64)
                * payment_frequency.clone());
    let dirty_K = K.clone()
        + (coupon_payment.clone()
            / payment_frequency.clone())
            * (T.clone()
                - time_of_last_payment.clone());

    // calculate sum of coupon payments I
    let I = map_multiple(
        move |cols| match cols {
            [b, c, d, e, f] => {
                let (b, c, d, e, f) = (b.clone(), c.clone(), d.clone(), e.clone(), f.clone());

                let time_to_next_payment = b.f64()?;
                let time_to_maturity = c.f64()?;
                let coupon_payment = d.f64()?;
                let payment_frequency = e.f64()?;
                let r_curve = f.list()?;

                let res: Float64Chunked = izip!(
                    time_to_next_payment,
                    time_to_maturity,
                    coupon_payment,
                    payment_frequency,
                    r_curve
                )
                .map(|(t1, tm, c, dt, r)| match (t1, tm, c, dt, r) {
                    (Some(t1), Some(tm), Some(c), Some(dt), Some(r)) => {
                        let r_vec: Vec<f64> = r.f64().unwrap().to_vec_null_aware().left().unwrap();
                        let num_payments = ((tm - t1) / dt).ceil() as usize;
                        Some(
                            (0..num_payments)
                                .map(|i| {
                                    let t = t1 + (i as f64) * dt;
                                    (-r_vec[i % r_vec.len()] * t).exp() * c
                                })
                                .sum::<f64>(),
                        )
                    }
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
            time_to_next_payment.clone(),
            T.clone(),
            coupon_payment.clone(),
            payment_frequency.clone(),
            r_curve.clone(),
        ],
        GetOutput::from_type(DataType::Float64),
    )
    .alias("I");

    let P = (-r_m.clone() * T.clone()).exp();

    let F_B =
        (B_0.clone() - I.clone()) / P.clone();

    println!("This is the forward bond price:");
    // let df = df.lazy().select([F_B.clone()]).collect()?;
    // println!("{df}");

    let d_1 = ((F_B.clone() / K.clone()).log(E)
        + sigma.clone().pow(2) * T.clone()
            / lit(2.0))
        / (sigma.clone() * T.clone().sqrt());
    let d_2 = d_1.clone()
        - sigma.clone() * T.clone().sqrt();

    let c = P.clone()
        * (F_B.clone() * N(d_1.clone())
            - K.clone() * N(d_2.clone()));
    let p = P.clone()
        * (K.clone() * N(-d_2.clone())
            - F_B.clone() * N(-d_1.clone()));

    let option_price = (when(
        option_type.clone().eq(lit("call")),
    )
    .then(c.clone())
    .otherwise(p.clone()))
    .alias("clean option price");

    // recalculate d1, d2, c, and p based on dirty_K
    let dirty_d_1 =
        ((F_B.clone() / dirty_K.clone()).log(E)
            + sigma.clone().pow(2) * T.clone()
                / lit(2.0))
            / (sigma.clone() * T.clone().sqrt());
    let dirty_d_2 = dirty_d_1.clone()
        - sigma.clone() * T.clone().sqrt();
    let dirty_c = P.clone()
        * (F_B.clone() * N(dirty_d_1.clone())
            - dirty_K.clone()
                * N(dirty_d_2.clone()));
    let dirty_p = P.clone()
        * (dirty_K.clone()
            * N(-dirty_d_2.clone())
            - F_B.clone()
                * N(-dirty_d_1.clone()));
    let dirty_option_price = (when(
        option_type.clone().eq(lit("call")),
    )
    .then(dirty_c.clone())
    .otherwise(dirty_p.clone()))
    .alias("dirty option price");

    let df = df
        .lazy()
        .select([
            name,
            option_price,
            dirty_option_price,
        ])
        .collect()?;
    Ok(df)
    // 10.1252 = cash
    // 12.4560 = quoted
    // 11.2522
}
