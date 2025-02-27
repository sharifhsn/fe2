use polars::prelude::*;
// use page 695 in Hull, Application 1 Revisited
pub fn a() -> PolarsResult<DataFrame> {
    let df = df!("T" => [7.0], "L" => [5_000_000.0], "yield curve" => [0.038], "sigma" => [0.10], "rho" => [1], )?;
    Ok(df)
}
