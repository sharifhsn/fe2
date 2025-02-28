use polars::prelude::*;
// use page 695 in Hull, Application 1 Revisited
pub fn a() -> PolarsResult<DataFrame> {
    // let mut df = df!("T*" => [7.0], "L" => [5_000_000.0], "yield curve" => [0.038], "sigma" => [0.10], "rho" => [1], "m" => [1.0])?;
    let df = df!(
        "T" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )?;
    let T = col("T");
    // all of these values are the same for each risk-free rate we are adjusting
    let Tstar = lit(7.0);
    let L = lit(5_000_000.0);
    let R_F = lit(0.038);
    // these are the same because the market variable is the interest rate
    let sigma_R = lit(0.10);
    let sigma_V = lit(0.10);
    // assume perfect correlation
    let rho = lit(1.0);
    // annual compounding
    let m = lit(1.0);

    let tau = Tstar.clone() - T.clone();

    let alpha_V = -(rho.clone()
        * sigma_V.clone()
        * sigma_R.clone()
        * R_F.clone()
        * tau.clone())
        / (lit(1.0) + R_F.clone() / m.clone());

    let R_adjusted = (R_F.clone()
        * (alpha_V.clone() * T.clone()).exp())
    .alias("R adjusted");

    let P = lit(1.0)
        / (lit(1.0) + R_F.clone())
            .pow(Tstar.clone());

    let V = (L.clone()
        * P.clone()
        * R_adjusted.clone().mean())
    .alias("Value after timing adjustment");

    let df = df
        .lazy()
        .select([R_adjusted, V])
        .collect()?;

    Ok(df)
}
