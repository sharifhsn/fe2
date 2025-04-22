use chrono::{format, Datelike, NaiveDate};
use nalgebra::{dmatrix, dvector, DMatrix, DVector};
use polars::prelude::*;
use time::macros::datetime;

use plotters::prelude::*;
use std::fs;
use std::path::Path;
use yahoo_finance_api::{self as yf, Decimal};
use yf::YahooConnector;

const TICKERS: [&str; 12] = [
    "AAPL", "SPY", "FXE", "EWJ", "GLD", "QQQ", "DBA", "USO", "XBI", "ILF", "EPP", "FEZ",
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
    let end = datetime!(2025-03-30 23:59 UTC);
    let provider = YahooConnector::new().unwrap();

    let mut tasks = Vec::new();

    for ticker in TICKERS {
        let provider = YahooConnector::new().unwrap();
        let task = tokio::spawn(async move {
            let resp = provider
                .get_quote_history_interval(ticker, start, end, "1d")
                .await
                .unwrap();
            let quotes = resp.quotes().unwrap();
            let mut df = struct_to_dataframe!(quotes, [adjclose]).unwrap();
            df.rename("adjclose", ticker.into()).unwrap();
            df
        });
        tasks.push(task);
    }
    let resp = provider
        .get_quote_history_interval("AAPL", start, end, "1d")
        .await
        .unwrap();
    let quotes = resp.quotes().unwrap();
    let mut combined_df = struct_to_dataframe!(quotes, [timestamp]).unwrap();

    let mut results = Vec::with_capacity(tasks.len());
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
        combined_df = combined_df.hstack(df.get_columns()).unwrap();
    }

    combined_df
}

#[tokio::main]

pub async fn a() -> PolarsResult<()> {
    let data_dir = "./data";
    let csv_path = format!("{}/midterm.csv", data_dir);

    let df = if !Path::new(&csv_path).exists() {
        // Ensure the data directory exists
        fs::create_dir_all(data_dir).expect("Unable to create data directory");
        let mut file = fs::File::create(&csv_path).expect("Unable to create file");

        // Load data and save to CSV
        let mut df = data_load().await;

        CsvWriter::new(&mut file).finish(&mut df)?;
        println!("Data saved to {}", csv_path);
        df
    } else {
        println!("CSV already exists at {}", csv_path);
        CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(csv_path.into()))?
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
    let df = lf.collect().expect("Failed to collect DataFrame");
    println!("DataFrame: {:?}", df);

    // Convert adjusted close prices into returns for all tickers
    let rets = TICKERS
        .iter()
        .map(|&ticker| ((col(ticker) / (col(ticker).shift(lit(1)))) - lit(1.0)).alias(ticker))
        .collect::<Vec<_>>();

    let df = df
        .lazy()
        .select(
            std::iter::once(col("timestamp").alias("date"))
                .chain(rets)
                .collect::<Vec<_>>(),
        )
        .drop_nulls(None)
        .collect()?;
    println!("Returns: {:?}", df);

    let hist_df = df
        .clone()
        .lazy()
        .filter(col("date").lt(datetime(DatetimeArgs::new(lit(2025), lit(1), lit(2)))))
        .collect()?;

    // how to turn into expected returns, covariance?
    let mu = hist_df
        .clone()
        .lazy()
        .select([col("^[A-Z]+$").mean() * lit(252.0)])
        .collect()?;

    // Turn this into a Nalgebra DVector
    // The mean returns are currently represented as a single row in a DataFrame
    // First convert that matrix into a vector.
    let mu_row = mu
        .get(0)
        .unwrap()
        .into_iter()
        .map(|x: AnyValue| {
            if let AnyValue::Float64(v) = x {
                v
            } else {
                panic!("Expected Float64 value");
            }
        })
        .collect::<Vec<_>>();
    // Now convert the vector into a DVector
    let mu = DVector::from_vec(mu_row);

    println!("Mu: {}", mu);

    let mut covariances = Vec::new();
    for &ticker1 in &TICKERS {
        for &ticker2 in &TICKERS {
            let cov = (((col(ticker1) - col(ticker1).mean())
                * (col(ticker2) - (col(ticker2).mean())))
            .mean()
                * lit(252.0)) // annualization
            .alias(format!("cov_{}_{}", ticker1, ticker2));
            covariances.push(cov);
        }
    }
    let cov = hist_df.clone().lazy().select(covariances).collect()?;

    // Now initialize a Nalgebra DMatrix with this covariance matrix
    // The covariance matrix is currently represented as a single row in a DataFrame
    // First convert that matrix into a vector.
    let cov_row = cov
        .get(0)
        .unwrap()
        .into_iter()
        .map(|x: AnyValue| {
            if let AnyValue::Float64(v) = x {
                v
            } else {
                panic!("Expected Float64 value");
            }
        })
        .collect::<Vec<_>>();
    // Now convert the vector into a DMatrix
    // The values are stored in a row-major order
    let n = TICKERS.len();
    let cov_matrix = DMatrix::from_vec(n, n, cov_row);
    println!("Covariance Matrix: {:.04}", cov_matrix);

    // Compute the betas of the securities in df with respect to the S&P-500 index (SPY)
    // Compute β_A, the beta of the stock AAPL with respect to SPY
    let spy_idx = 1;
    let betas = (0..TICKERS.len())
        .map(|i| {
            let cov = cov_matrix[(i, spy_idx)];
            let var = cov_matrix[(spy_idx, spy_idx)];
            cov / var
        })
        .collect::<Vec<_>>();
    // Convert betas into a DVector
    let betas = DVector::from_vec(betas);

    let beta_A = betas[0];
    println!("AAPL Beta: {}", beta_A);
    let betas_wA = betas.clone().remove_row(0);
    println!("Betas: {}", betas_wA);
    let cov_matrix_wA = cov_matrix.clone().remove_column(0).remove_row(0);
    // Find the portfolio w_mV(β_A)
    // Compute the weights of the portfolio that minimizes the variance
    // Solve constrained minimization with Lagrange multipliers
    let ones = DVector::from_element(n - 1, 1.0);
    let invcov = cov_matrix_wA.try_inverse().unwrap();
    let A = (ones.clone().transpose() * invcov.clone() * ones.clone())[(0, 0)];
    let B = (ones.clone().transpose() * invcov.clone() * betas_wA.clone())[(0, 0)];
    let C = (betas_wA.clone().transpose() * invcov.clone() * betas_wA.clone())[(0, 0)];

    let ABC = dmatrix![
        A, B;
        B, C
    ];
    let invABC = ABC.try_inverse().unwrap();
    let lagrange = invABC * dvector![2.0, 2.0 * beta_A];
    let weights_hedged = 0.5 * invcov * (lagrange[0] * betas_wA + lagrange[1] * ones);
    println!("Weights: {}", weights_hedged);
    // I can use this portfolio to hedge my position in AAPL by shorting it.
    // The expected return of this strategy is
    let mu_wA = mu.clone().remove_row(0);
    let expected_return_hedged = mu[0] - (mu_wA.transpose() * weights_hedged.clone())[(0, 0)];
    println!("Expected Return: {}", expected_return_hedged);

    // now try the enlarged investment universe with target beta 0
    let ones = DVector::from_element(n, 1.0);
    let invcov = cov_matrix.try_inverse().unwrap();
    let A = (ones.clone().transpose() * invcov.clone() * ones.clone())[(0, 0)];
    let B = (ones.clone().transpose() * invcov.clone() * betas.clone())[(0, 0)];
    let C = (betas.clone().transpose() * invcov.clone() * betas.clone())[(0, 0)];
    let ABC = dmatrix![
        A, B;
        B, C
    ];
    let invABC = ABC.try_inverse().unwrap();
    let lagrange = invABC * dvector![2.0, 0.0];
    let weights_neutral = 0.5 * invcov * (lagrange[0] * betas + lagrange[1] * ones);
    println!("Weights: {}", weights_neutral);
    // expected return of this strategy
    let expected_return_neutral = mu.transpose() * weights_neutral.clone();
    println!("Expected Return: {}", expected_return_neutral);

    // Compute the realized daily returns of the portfolios built by
    // the two weights for the period from Jan 3 2025 to Mar 30 2025
    let realized_df = df
        .clone()
        .lazy()
        .filter(col("date").gt(datetime(DatetimeArgs::new(lit(2025), lit(1), lit(2)))))
        .collect()?;

    // Convert this df into a bunch of vectors.
    let binding = realized_df
        .clone()
        .column("date")?
        .date()?
        .strftime("%Y-%d-%m")?;
    let times: Vec<Option<&str>> = binding.iter().collect();
    let times = times.into_iter().map(|x| x.unwrap()).collect::<Vec<_>>();
    let times = times
        .into_iter()
        .map(|t| chrono::NaiveDate::parse_from_str(t, "%Y-%d-%m").unwrap())
        .collect::<Vec<_>>();

    let mut tickers = vec![];
    for ticker in TICKERS {
        let ticker = df.column(ticker)?.f64()?;
        let ticker = ticker.to_vec_null_aware().left().unwrap();
        tickers.push(ticker);
    }
    let mut hedged_rets = vec![];
    let mut neutral_rets = vec![];

    let num_rows = tickers[0].len();
    for i in 0..num_rows {
        let row = tickers.iter().map(|v| v[i]).collect::<Vec<_>>();
        let hedged_ret = row[0]
            - row[1..]
                .iter()
                .zip(weights_hedged.iter())
                .map(|(&x, &w)| x * w)
                .sum::<f64>();
        hedged_rets.push(hedged_ret);
        let neutral_ret = row
            .iter()
            .zip(weights_neutral.iter())
            .map(|(&x, &w)| x * w)
            .sum::<f64>();
        neutral_rets.push(neutral_ret);
    }

    let realized_mu = realized_df
        .clone()
        .lazy()
        .select([col("^[A-Z]+$").mean() * lit(252.0)])
        .collect()?;

    let realized_mu_row = realized_mu
        .get(0)
        .unwrap()
        .into_iter()
        .map(|x: AnyValue| {
            if let AnyValue::Float64(v) = x {
                v
            } else {
                panic!("Expected Float64 value");
            }
        })
        .collect::<Vec<_>>();
    // Now convert the vector into a DVector
    let realized_mu = DVector::from_vec(realized_mu_row);

    let realized_mu_wA = realized_mu.clone().remove_row(0);

    let mut covariances = Vec::new();
    for &ticker1 in &TICKERS {
        for &ticker2 in &TICKERS {
            let cov = (((col(ticker1) - col(ticker1).mean())
                * (col(ticker2) - (col(ticker2).mean())))
            .mean()
                * lit(252.0)) // annualization
            .alias(format!("cov_{}_{}", ticker1, ticker2));
            covariances.push(cov);
        }
    }
    let realized_cov = realized_df.clone().lazy().select(covariances).collect()?;

    let cov_row = realized_cov
        .get(0)
        .unwrap()
        .into_iter()
        .map(|x: AnyValue| {
            if let AnyValue::Float64(v) = x {
                v
            } else {
                panic!("Expected Float64 value");
            }
        })
        .collect::<Vec<_>>();
    let n = TICKERS.len();
    let realized_cov_matrix = DMatrix::from_vec(n, n, cov_row);
    let realized_cov_matrix_wA = realized_cov_matrix.clone().remove_column(0).remove_row(0);
    println!("Covariance Matrix: {:.04}", realized_cov_matrix);

    // Compare weights_hedged and weights_neutral performance
    let hedged_ret = realized_mu[0] - (realized_mu_wA.transpose() * weights_hedged.clone())[(0, 0)];
    let neutral_ret = (realized_mu.transpose() * weights_neutral.clone())[(0, 0)];
    println!(
        "Comparison: hedged_ret: {}, neutral_ret: {}",
        hedged_ret, neutral_ret
    );

    // For hedged_rets and neutral_rets, calculate stdev
    let hedged_rets_stdev = hedged_rets
        .iter()
        .map(|&x| (x - hedged_ret).powi(2))
        .sum::<f64>()
        / (hedged_rets.len() as f64 - 1.0);
    let hedged_rets_stdev = hedged_rets_stdev.sqrt();
    println!("Hedged Returns Stdev: {}", hedged_rets_stdev);
    let neutral_rets_stdev = neutral_rets
        .iter()
        .map(|&x| (x - neutral_ret).powi(2))
        .sum::<f64>()
        / (neutral_rets.len() as f64 - 1.0);
    let neutral_rets_stdev = neutral_rets_stdev.sqrt();
    println!("Neutral Returns Stdev: {}", neutral_rets_stdev);

    // Print the plot of the time series using hedged_rets and neutral_rets with times
    let root = BitMapBackend::new("returns_plot.png", (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_ret = hedged_rets
        .iter()
        .chain(neutral_rets.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_ret = hedged_rets
        .iter()
        .chain(neutral_rets.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Hedged vs Neutral Returns", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(times[0]..times[times.len() - 1], min_ret..max_ret)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| x.to_string())
        .y_label_formatter(&|y| format!("{:.2}", y))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(hedged_rets.iter()).map(|(t, r)| (*t, *r)),
            &RED,
        ))
        .unwrap()
        .label("Hedged Returns")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(neutral_rets.iter()).map(|(t, r)| (*t, *r)),
            &BLUE,
        ))
        .unwrap()
        .label("Neutral Returns")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    println!("Plot saved to returns_plot.png");

    // Cumulative return
    let cum_hedged = hedged_rets.iter().fold(Vec::<f64>::new(), |mut acc, &x| {
        acc.push(acc.last().unwrap_or(&1.0) * (1.0 + x));
        acc
    });
    let cum_neutral = neutral_rets.iter().fold(Vec::<f64>::new(), |mut acc, &x| {
        acc.push(acc.last().unwrap_or(&1.0) * (1.0 + x));
        acc
    });

    // Plot the cumulative returns
    let root = BitMapBackend::new("cumulative_returns_plot.png", (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_cum_ret = cum_hedged
        .iter()
        .chain(cum_neutral.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_cum_ret = cum_hedged
        .iter()
        .chain(cum_neutral.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Cumulative Hedged vs Neutral Returns",
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(times[0]..times[times.len() - 1], min_cum_ret..max_cum_ret)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| x.to_string())
        .y_label_formatter(&|y| format!("{:.2}", y))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(cum_hedged.iter()).map(|(t, r)| (*t, *r)),
            &RED,
        ))
        .unwrap()
        .label("Cumulative Hedged Returns")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(cum_neutral.iter()).map(|(t, r)| (*t, *r)),
            &BLUE,
        ))
        .unwrap()
        .label("Cumulative Neutral Returns")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    println!("Cumulative returns plot saved to cumulative_returns_plot.png");

    println!("Hedged returns: {:?}", hedged_rets);
    println!("Neutral returns: {:?}", neutral_rets);
    Ok(())
}
