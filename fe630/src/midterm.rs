use chrono::{Datelike, NaiveDate};
use polars::prelude::*;
use time::macros::datetime;

use yahoo_finance_api::{self as yf, Decimal};

use std::fs;
use std::path::Path;
use yf::YahooConnector;

const TICKERS: [PlSmallStr; 13] = [
    PlSmallStr::from_static("AAPL"),
    PlSmallStr::from_static("SPY"),
    PlSmallStr::from_static("FXE"),
    PlSmallStr::from_static("EWJ"),
    PlSmallStr::from_static("GLD"),
    PlSmallStr::from_static("QQQ"),
    PlSmallStr::from_static("SHV"),
    PlSmallStr::from_static("DBA"),
    PlSmallStr::from_static("USO"),
    PlSmallStr::from_static("XBI"),
    PlSmallStr::from_static("ILF"),
    PlSmallStr::from_static("EPP"),
    PlSmallStr::from_static("FEZ"),
];

macro_rules! struct_to_dataframe {
    ($input:expr, [$($field:ident),+]) => {
        {
            let len = $input.len().to_owned();

            $(let mut $field = Vec::with_capacity(len);)*

            for e in $input.into_iter() {
                $($field.push(e.$field.clone());)*
            }
            df! {
                $(stringify!($field) => $field,)*
            }
        }
    };
}

pub struct Quote {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub volume: u64,
    pub close: f64,
    pub adjclose: f64,
}

pub async fn data_load() -> DataFrame {
    let start = datetime!(2024-01-01 00:00 UTC);
    let end = datetime!(2025-02-28 23:59 UTC);
    let provider = YahooConnector::new().unwrap();

    let mut tasks = Vec::new();

    for ticker in TICKERS {
        let provider =
            YahooConnector::new().unwrap();
        let task = tokio::spawn(async move {
            let resp = provider
                .get_quote_history_interval(
                    &ticker, start, end, "1d",
                )
                .await
                .unwrap();
            let quotes = resp.quotes().unwrap();
            let mut df = struct_to_dataframe!(
                quotes,
                [adjclose]
            )
            .unwrap();
            df.rename("adjclose", ticker.into())
                .unwrap();
            df
        });
        tasks.push(task);
    }
    let resp = provider
        .get_quote_history_interval(
            "AAPL", start, end, "1d",
        )
        .await
        .unwrap();
    let quotes = resp.quotes().unwrap();
    let mut combined_df =
        struct_to_dataframe!(quotes, [timestamp])
            .unwrap();

    let mut results =
        Vec::with_capacity(tasks.len());
    for task in tasks {
        match task.await {
            Ok(result) => results.push(result),
            Err(e) => {
                eprintln!("Error: {:?}", e);
            }
        }
    }

    for df in results {
        println!("df: {:?}", df);
        combined_df = combined_df
            .hstack(df.get_columns())
            .unwrap();
    }

    combined_df
}

#[tokio::main]

pub async fn a() -> PolarsResult<()> {
    let data_dir = "./data";
    let csv_path =
        format!("{}/midterm.csv", data_dir);

    let df = if !Path::new(&csv_path).exists() {
        // Ensure the data directory exists
        fs::create_dir_all(data_dir).expect(
            "Unable to create data directory",
        );
        let mut file =
            fs::File::create(&csv_path)
                .expect("Unable to create file");

        // Load data and save to CSV
        let mut df = data_load().await;

        CsvWriter::new(&mut file)
            .finish(&mut df)?;
        println!("Data saved to {}", csv_path);
        df
    } else {
        println!(
            "CSV already exists at {}",
            csv_path
        );
        CsvReadOptions::default()
            .try_into_reader_with_file_path(
                Some(csv_path.into()),
            )?
            .finish()?
    };

    // Convert timestamp to milliseconds
    // and cast to Datetime
    let lf = df.lazy().with_column(
        (col("timestamp") * lit(1000))
            .cast(DataType::Datetime(
                TimeUnit::Milliseconds,
                Some("UTC".into()),
            ))
            .cast(DataType::Date),
    );
    let df = lf
        .collect()
        .expect("Failed to collect DataFrame");
    println!("DataFrame: {:?}", df);

    // Convert adjusted close prices into returns for all tickers
    let rets = TICKERS
        .iter()
        .map(|ticker| {
            ((col(ticker.clone())
                / (col(ticker.clone())
                    .shift(lit(1))))
                - lit(1.0))
            .alias(format!("{}_returns", ticker))
        })
        .collect::<Vec<_>>();

    let df = df
        .lazy()
        .select(
            std::iter::once(
                col("timestamp")
                    .cast(DataType::Int64)
                    .alias("date"),
            )
            .chain(rets)
            .collect::<Vec<_>>(),
        )
        .drop_nulls(None)
        .collect()?;

    // let returns

    // how to turn into expected variance, covariance?

    println!("Returns: {:?}", df);
    Ok(())
}
