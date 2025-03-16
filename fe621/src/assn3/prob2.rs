use std::path::PathBuf;

use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::assn3::prob1::FiniteDifference;

pub fn a() -> PolarsResult<()> {
    let df_hist = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from("historical_data.csv")))?
        .finish()?;
    //println!("{:?}", df_hist);
    const R: f64 = 0.0433;
    let data2 = df_hist
        .lazy()
        .filter(
            col("Datetime")
                .str()
                .to_date(StrptimeOptions {
                    format: Some("%Y-%m-%d %H:%M:%S%z".into()),
                    strict: false,
                    exact: true,
                    ..Default::default()
                })
                .dt()
                .date()
                .eq(datetime(DatetimeArgs::new(lit(2025), lit(2), lit(14)))
                    .dt()
                    .date()),
        )
        .collect()?;
    let data2_latest = data2.sort(["Datetime"], Default::default())?.tail(Some(1));
    let s0_nvda = data2_latest.column("NVDA")?.f64()?.get(0).unwrap();

    let mut df_opt = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from("options_data.csv")))?
        .finish()?
        .lazy()
        .with_column(
            col("lastTradeDate")
                .str()
                .to_date(StrptimeOptions {
                    format: Some("%Y-%m-%d %H:%M:%S%z".into()),
                    strict: false,
                    exact: true,
                    ..Default::default()
                })
                .alias("lastTradeDate"),
        )
        .with_column(
            col("expirationDate")
                .str()
                .to_date(StrptimeOptions {
                    format: Some("%Y-%m-%d".into()),
                    strict: false,
                    exact: true,
                    ..Default::default()
                })
                .alias("expirationDate"),
        )
        .with_column(
            (col("expirationDate").dt().date() - col("lastTradeDate").dt().date())
                .alias("timeToMaturity"),
        )
        // Filter for NVDA calls and puts
        .filter(col("ticker").eq(lit("NVDA")))
        .with_column((col("strike") - lit(s0_nvda)).abs().alias("abs_diff"))
        .group_by(["expirationDate", "optionType"])
        .agg([col("*")
            .sort_by([col("abs_diff")], SortMultipleOptions::default())
            .head(Some(10))])
        .select([
            col("strike"),
            col("lastPrice"),
            col("bid"),
            col("ask"),
            col("impliedVolatility"),
            col("optionType"),
            col("expirationDate"),
            col("timeToMaturity"),
        ])
        .explode([
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "impliedVolatility",
            "timeToMaturity",
        ])
        .sort(
            ["expirationDate", "optionType", "strike"],
            Default::default(),
        )
        .collect()?;

    println!("{:?}", df_opt);

    // Extracting data

    Ok(())
}
