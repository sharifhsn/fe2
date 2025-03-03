use std::f64::consts::E;

use polars::prelude::*;

pub fn a() -> PolarsResult<DataFrame> {
    let df = df!(
        // yield curve is assumed flat
        "r" => [0.038],
        "m" => [E],
        // time of payoff
        "T" => [4.0],
        // time of first forward rate
        "T_1" => [4.0 + 2.0],
        // time of second forward rate
        "T_2" => [4.0 + 6.0],
        "L" => [5_000_000.0],
        "sigma" => [0.15],
    )?;
    let r = col("r");
    let m = col("m");
    let T = col("T");
    let T_1 = col("T_1");
    let T_2 = col("T_2");
    let L = col("L");
    let sigma = col("sigma");

    let tau_1 = T_1.clone() - T.clone();
    let tau_2 = T_2.clone() - T.clone();

    let P = when(m.clone().eq(E))
        .then(
            // use continuous compounding
            (-r.clone() * T.clone()).exp(),
        )
        .otherwise(
            // else discrete compounding
            lit(1.0)
                / (lit(1.0)
                    + r.clone() * m.clone())
                .pow(T.clone() / m.clone()),
        );
    // ((lit(-1.0)
    // + tau_2.clone().pow(2))
    // / tau_2.clone())
    let conv1 = r.clone()
        - r.clone().pow(2)
            * sigma.clone().pow(2)
            * T.clone()
            * (-tau_1.clone())
            / lit(2.0);
    let conv2 = r.clone()
        - r.clone().pow(2)
            * sigma.clone().pow(2)
            * T.clone()
            * (-tau_2.clone())
            / lit(2.0);

    let convexity_1: Expr = (r.clone().pow(2)
        * sigma.clone().pow(2)
        * tau_1.clone()
        * T.clone())
        / (lit(1.0) + r.clone() * tau_1.clone());

    let convexity_2: Expr = (r.clone().pow(2)
        * sigma.clone().pow(2)
        * tau_2.clone()
        * T.clone())
        / (lit(1.0) + r.clone() * tau_2.clone());

    // yield curve is flat, so the r term is elided
    let V = (P.clone()
        * L.clone()
        * (conv1.clone() - conv2.clone()))
    .alias("Value after convexity adjustment");
    let df = df.lazy().select([V]).collect()?;
    Ok(df)
}
