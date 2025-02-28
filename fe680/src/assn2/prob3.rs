use std::f64::consts::E;

use polars::prelude::*;

use crate::util::N;
// calculate caplet/floorlet
pub fn a() -> PolarsResult<DataFrame> {
    let df = df!(
        "start of collar" => [0.0],
        "collar length" => [5.0],
        // assumed flat
        "r" => [0.035],
        "m" => [E],
        // assumes constant tenor
        "tenor" => [0.25],
        "R_F" => [0.031],
        "R_C" => [0.038],
        "L" => [5_000_000.0],
        "sigma" => [0.12],
    )?;
    let start_of_collar = col("start of collar");
    let collar_length = col("collar length");
    let r = col("r");
    let m = col("m");

    let tenor = col("tenor");
    let R_F = col("R_F");
    let R_C = col("R_C");
    let L = col("L");
    let sigma = col("sigma");

    // reset times
    let T = linear_space(
        start_of_collar.clone(),
        start_of_collar.clone()
            + collar_length.clone(),
        collar_length.clone() / tenor.clone(),
        ClosedInterval::Right,
    )
    .alias("T");

    // payment times
    let T1 =
        (T.clone() + tenor.clone()).alias("T1");

    let P = when(m.clone().eq(E))
        .then(
            // use continuous compounding
            (-r.clone() * T1.clone()).exp(),
        )
        .otherwise(
            // else discrete compounding
            lit(1.0)
                / (lit(1.0)
                    + r.clone() * m.clone())
                .pow(T1.clone() / m.clone()),
        );

    let F = ((r.clone() * tenor.clone()).exp()
        - lit(1.0))
        / tenor.clone();

    let d_1_C = ((F.clone() / R_C.clone()
        - lit(1.0))
    .log1p()
        + sigma.clone().pow(2) * T.clone()
            / lit(2.0))
        / (sigma.clone() * T.clone().sqrt());

    let d_2_C = d_1_C.clone()
        - sigma.clone() * T.clone().sqrt();

    let d_1_F = ((F.clone() / R_F.clone()
        - lit(1.0))
    .log1p()
        + sigma.clone().pow(2) * T.clone()
            / lit(2.0))
        / (sigma.clone() * T.clone().sqrt());

    let d_2_F = d_1_F.clone()
        - sigma.clone() * T.clone().sqrt();

    let cap = P.clone()
        * tenor.clone()
        * L.clone()
        * (F.clone() * N(d_1_C.clone())
            - R_C.clone() * N(d_2_C.clone()));

    let floor = P.clone()
        * tenor.clone()
        * L.clone()
        * (R_F.clone() * N(-d_2_F.clone())
            - F.clone() * N(-d_1_F.clone()));

    let V = (cap.clone() - floor.clone())
        .sum()
        .alias("Value of collar");

    let df = df.lazy().select([V]).collect()?;
    Ok(df)
}
