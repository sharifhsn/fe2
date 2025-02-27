use crate::util::N;
use itertools::izip;
use polars::prelude::*;

pub fn a() -> PolarsResult<DataFrame> {
    let df = df!(
        "name" => ["example", "problem 2"],
        // assumes that yield curve is flat with continuous compounding
        "yield curve" => [0.06, 0.0405],
        "L" => [100_000_000.0, 5_000_000.0],
        "n" => [3.0, 5.0],
        "sigma" => [0.20, 0.15],
        "m" => [0.5, 1.0],
        "T" => [5.0, 2.0],
        "s_k" => [0.062, 0.0415],
        "yield curve compounding" => ["continuous", "annual"],
    )?;
    let name = col("name");
    let yield_curve = col("yield curve");
    let L = col("L");
    let n = col("n");
    let sigma = col("sigma");
    let m = col("m");
    let T = col("T");
    let s_k = col("s_k");
    let yield_curve_compounding =
        col("yield curve compounding");

    let s_0 = when(
        yield_curve_compounding
            .clone()
            .eq(lit("continuous")),
    )
    .then(
        ((yield_curve.clone() * m.clone()).exp()
            - lit(1.0))
            / m.clone(),
    )
    .otherwise(yield_curve.clone());

    let A = map_multiple(
        move |cols| match cols {
            [b, c, d, e, f] => {
                let (b, c, d, e, f) = (
                    b.clone(),
                    c.clone(),
                    d.clone(),
                    e.clone(),
                    f.clone(),
                );

                let T = b.f64()?;
                let n = c.f64()?;
                let m =
                    d.f64()?;
                let yield_curve = e.f64()?;
                let yield_curve_compounding = f.str()?;

                let res: Float64Chunked = izip!(
                    T,
                    n,
                    m,
                    yield_curve,
                    yield_curve_compounding,
                )
                .map(|(T, n, m, r, s)| {
                    match (T, n, m, r, s) {
                        (
                            Some(T),
                            Some(n),
                            Some(m),
                            Some(r),
                            Some(s)
                        ) => {
                            let num_payments =
                                (n / m).floor() as usize;
                            Some(m *
                                (1..=num_payments)
                                    .map(|i| {
                                        let t = T + m * (i as f64);
                                        if s == "continuous" {
                                            (-r * t)
                                            .exp()
                                        } else {
                                            1.0 / (1.0 + r).powf(t)
                                        }
                                    })
                                    .sum::<f64>(),
                            )
                        }
                        _ => None,
                    }
                })
                .collect();

                Ok(Some(res.into_column()))
            }
            _ => Err(PolarsError::ComputeError(
                "Expected exactly 5 columns"
                    .into(),
            )),
        },
        &[
            T.clone(),
            n.clone(),
            m.clone(),
            yield_curve.clone(),
            yield_curve_compounding.clone(),
        ],
        GetOutput::from_type(DataType::Float64),
    )
    .alias("A");

    let d_1 = (((s_0.clone() / s_k.clone()
        - lit(1.0))
    .log1p()
        + sigma.clone().pow(2) * T.clone()
            / lit(2.0))
        / (sigma.clone() * T.clone().sqrt()))
    .alias("d_1");
    let d_2 = (d_1.clone()
        - sigma.clone() * T.clone().sqrt())
    .alias("d_2");

    let swaption_price = (L.clone()
        * A.clone()
        * (s_0.clone() * N(d_1.clone())
            - s_k.clone() * N(d_2.clone())))
    .alias("swaption price");

    let df = df
        .lazy()
        .select([
            name,
            A,
            d_1,
            d_2,
            swaption_price,
        ])
        .collect()?;
    Ok(df)
}
