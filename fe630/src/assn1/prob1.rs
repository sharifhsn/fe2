use nalgebra::{dmatrix, dvector, DMatrix, DVector};
use polars::prelude::*;
use std::f64::consts::E;

/// Utility function
pub fn U(e: Expr) -> Expr {
    e.log(E)
}

/// Inverse utility function
pub fn U_inv(e: Expr) -> Expr {
    e.exp()
}

pub fn U_k(e: Expr, k: Expr) -> Expr {
    e.pow(lit(1.0) / k)
}

pub fn a() -> PolarsResult<DataFrame> {
    let mut df = df!(
        "W_0" => [1000.0],
        "p_w" => [0.5],
        "p_l" => [0.5],
        "w" => [200.0],
        "l" => [200.0]
    )?;
    let W_0 = col("W_0");
    let p_w = col("p_w");
    let p_l = col("p_l");
    let w = col("w");
    let l = col("l");

    // let u = W.clone().log(E);
    let EUW_T = (p_w.clone() * U(W_0.clone() + w.clone())
        + p_l.clone() * U(W_0.clone() - l.clone()))
    .alias("\\mathbb{E}[U(W_T)]");

    let CE = (U_inv(EUW_T.clone())).alias("CE");

    let var = p_w.clone() * w.clone().pow(2) + p_l.clone() * l.clone().pow(2);

    let EW_T = (p_w.clone() * (W_0.clone() + w.clone()) + p_l.clone() * (W_0.clone() - l.clone()))
        .alias("\\mathbb{E}[W_T]");
    let RP = (EW_T.clone() - CE.clone()).alias("RP");

    let Ez = p_w.clone() * w.clone() - p_l.clone() * l.clone();

    let W_Tstar = EW_T.clone() + Ez.clone();

    let A = (-lit(1.0) / W_Tstar.clone());

    let y = -var / lit(2.0) * A;

    df.with_column(Column::new("W_0".into(), [1200.0]))?;

    let df = df.lazy().select([CE]).collect()?;

    Ok(df)
}

pub fn b() -> PolarsResult<DataFrame> {
    let mut df = df!(
        "W_0" => [1000.0],
        "p_w" => [2.0/3.0],
        "p_l" => [1.0/3.0],
        "w" => [205],
        "l" => [400],
        "k" => [0.5],
    )?;
    let W_0 = col("W_0");
    let p_w = col("p_w");
    let p_l = col("p_l");
    let w = col("w");
    let l = col("l");
    let k = col("k");

    let A = -(k.clone() - lit(1.0)) / W_0.clone();

    let risk_attitude = (when(A.clone().gt(0)).then(lit("risk-averse")).otherwise(
        when(A.clone().lt(0))
            .then(lit("risk-taking"))
            .otherwise(lit("risk-neutral")),
    ))
    .alias("risk attitude");

    Ok(df)
}
pub fn c() {
    let n = 3;
    let lambda = 0.5;

    let mu: DVector<f64> = dvector![0.08, 0.12, 0.15];
    let sigma: DMatrix<f64> = dmatrix![
        0.02,  0.01,  0.015;
        0.01,  0.03,  0.02;
        0.015, 0.02,  0.04
    ];
    let w = dvector![0.3, 0.4, 0.3];

    let Er = w.transpose() * mu;
}

pub fn U_r(r: f64, lambda: f64) -> f64 {
    (lambda * r).ln_1p()
}
pub fn U_approx(r: f64, lambda: f64) -> f64 {
    lambda * r - lambda.powi(2) * r.powi(2) / 2.0
}
