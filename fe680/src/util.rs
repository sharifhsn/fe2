use polars::prelude::*;
use statrs::distribution::ContinuousCDF;

// polars does not natively provide the ability to get the normal cdf,
// so we have to use an alternative library and use arbitrary mapping
pub fn N(e: Expr) -> Expr {
    e.map(
        |s| {
            Ok(Some(Column::new(
                "".into(),
                s.f64()
                    .unwrap()
                    .into_iter()
                    .map(|ca| {
                        ca.map(|f| {
                            statrs::distribution::Normal::standard()
                                .cdf(f)
                        })
                    })
                    .collect::<Vec<_>>(),
            )))
        },
        GetOutput::from_type(DataType::Float64),
    )
}
