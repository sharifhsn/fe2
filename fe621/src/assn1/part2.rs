use polars::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::E;
use std::path::PathBuf;

#[derive(Clone, Copy)]
enum OptionType {
    Call,
    Put,
}

fn black_scholes(option_type: OptionType, s0: f64, σ: f64, τ: f64, k: f64, r: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let d1 = (s0.ln() - k.ln() + (r + 0.5 * σ.powi(2)) * τ) / (σ * τ.sqrt());
    let d2 = d1 - σ * τ.sqrt();

    match option_type {
        OptionType::Call => s0 * normal.cdf(d1) - k * E.powf(-r * τ) * normal.cdf(d2),
        OptionType::Put => k * E.powf(-r * τ) * normal.cdf(-d2) - s0 * normal.cdf(-d1),
    }
}

pub trait Rootable {
    fn secant(&self, x0: f64, x1: f64, ε: f64, max_iter: usize) -> Option<f64>;
    fn bisect(&self, a: f64, b: f64, ε: f64, max_iter: usize) -> Option<f64>;
}
impl<F> Rootable for F
where
    F: Fn(f64) -> f64,
{
    fn bisect(&self, mut a: f64, mut b: f64, ε: f64, max_iter: usize) -> Option<f64> {
        if (self(a) * self(b)).is_sign_positive() {
            eprintln!("f(a) and f(b) must have opposite signs");
            return None;
        }

        let mut mid;
        for _ in 0..max_iter {
            mid = (a + b) / 2.0;

            if self(mid).abs() < ε {
                return Some(mid);
            }

            if (self(mid) * self(a)).is_sign_negative() {
                b = mid;
            } else {
                a = mid;
            }
        }
        eprintln!("Maximum iterations reached without finding the root.");
        None
    }

    fn secant(&self, mut x0: f64, mut x1: f64, ε: f64, max_iter: usize) -> Option<f64> {
        for _ in 0..max_iter {
            let f0 = self(x0);
            let f1 = self(x1);

            if (f1 - f0).abs() < 1e-10 {
                eprintln!("Denominator too small, Secant method failed.");
                return None;
            }

            let x_new = x1 - f1 * (x1 - x0) / (f1 - f0);

            if (x_new - x1).abs() < ε {
                return Some(x_new);
            }

            x0 = x1;
            x1 = x_new;
        }

        eprintln!("Maximum iterations reached without finding the root.");
        None
    }
}

const TOL: f64 = 1e-5;
const R: f64 = 0.0433;

fn process_options(df: DataFrame, s0: f64, ticker: &str) -> PolarsResult<DataFrame> {
    let df = df.lazy().filter(col("ticker").eq(lit(ticker))).collect()?;
    let expiration_dates = df.column("expirationDate")?.date()?.unique()?;
    let v: Vec<_> = expiration_dates
        .into_iter()
        .map(|expiration_date| -> PolarsResult<LazyFrame> {
            let options_for_expiry = df
                .clone()
                .lazy()
                .filter(col("expirationDate").eq(lit(expiration_date.unwrap())))
                .collect()?;
            let atm_call_option = options_for_expiry
                .clone()
                .lazy()
                .filter(
                    col("inTheMoney")
                        .eq(lit(true))
                        .and(col("optionType").eq(lit("call"))),
                )
                .sort(["strike"], Default::default())
                .tail(1)
                .collect()?;
            let atm_put_option = options_for_expiry
                .clone()
                .lazy()
                .filter(
                    col("inTheMoney")
                        .eq(lit(true))
                        .and(col("optionType").eq(lit("put"))),
                )
                .sort(
                    ["strike"],
                    SortMultipleOptions::default().with_order_descending(true),
                )
                .tail(1)
                .collect()?;

            let opt_calc = |opt_type: OptionType| -> PolarsResult<f64> {
                let atm_option = match opt_type {
                    OptionType::Call => atm_call_option.clone(),
                    OptionType::Put => atm_put_option.clone(),
                };
                println!("{:?}", atm_option);
                let k = atm_option.column("strike")?.f64()?.get(0).unwrap();
                let bid = atm_option.column("bid")?.f64()?.get(0).unwrap();
                let ask = atm_option.column("ask")?.f64()?.get(0).unwrap();

                let last_trade_date = atm_option.column("lastTradeDate")?.date()?.get(0).unwrap();
                let expiration_date = atm_option.column("expirationDate")?.date()?.get(0).unwrap();

                let market_price = (bid + ask) / 2.0;
                let tau = (expiration_date - last_trade_date) as f64 / 252.0;
                let implied_volatility = |sigma: f64| {
                    let bs_price = black_scholes(opt_type, s0, sigma, tau, k, R);
                    bs_price - market_price
                };
                let iv = implied_volatility.bisect(0.0001, 2.0, TOL, 100);

                Ok(iv.unwrap())
            };
            let calc_call_iv = opt_calc(OptionType::Call)?;
            let calc_put_iv = opt_calc(OptionType::Put)?;

            // Get all options between in the money and out of the money, using 0.95 and 1.05 as the boundary.
            // Moneyness is defined here by the ratio of S0 to the strike price.
            let between_call_options = options_for_expiry
                .clone()
                .lazy()
                .filter(
                    lit(s0)
                        .gt(lit(0.95) * col("strike"))
                        .and(lit(s0).lt(lit(1.05) * col("strike")))
                        .and(col("optionType").eq(lit("call"))),
                )
                .collect()?;
            let between_put_options = options_for_expiry
                .clone()
                .lazy()
                .filter(
                    lit(s0)
                        .gt(lit(0.95) * col("strike"))
                        .and(lit(s0).lt(lit(1.05) * col("strike")))
                        .and(col("optionType").eq(lit("put"))),
                )
                .collect()?;
            // Average the implied volatilities of these options
            let avg_call_iv = between_call_options
                .column("impliedVolatility")?
                .f64()?
                .mean()
                .unwrap();
            let avg_put_iv = between_put_options
                .column("impliedVolatility")?
                .f64()?
                .mean()
                .unwrap();

            // Compare average implied volatility to the calculated implied volatility
            println!("Call:");
            println!("Average Implied Vol: {:?}", avg_call_iv);
            println!("Calculated Implied Vol: {:?}", calc_call_iv);
            println!("Put:");
            println!("Average Implied Vol: {:?}", avg_put_iv);
            println!("Calculated Implied Vol: {:?}", calc_put_iv);

            Ok(df!(
                "expirationDate" => [expiration_date],
                "ticker" => [ticker],
                "callAvgIV" => [avg_call_iv],
                "callCalcIV" => [calc_call_iv],
                "putAvgIV" => [avg_put_iv],
                "putCalcIV" => [calc_put_iv]
            )?
            .lazy())
        })
        .filter_map(Result::ok)
        .collect();
    let full = concat(v, Default::default())?;
    full.collect()
}

pub fn a() -> PolarsResult<DataFrame> {
    let df_hist = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from("historical_data.csv")))?
        .finish()?;
    //println!("{:?}", df_hist);
    let df_opt = CsvReadOptions::default()
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
        .collect()?;

    let latest_prices = df_hist
        .sort(["Datetime"], Default::default())?
        .head(Some(1));
    let data1 = df_hist
        .clone()
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
                .eq(datetime(DatetimeArgs::new(lit(2025), lit(2), lit(13)))
                    .dt()
                    .date()),
        )
        .collect()?;
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
    let data1_latest = data1.sort(["Datetime"], Default::default())?.tail(Some(1));
    let data2_latest = data2.sort(["Datetime"], Default::default())?.tail(Some(1));
    // println!("{:?}", data1_latest);
    let s0_nvda = data2_latest.column("NVDA")?.f64()?.get(0).unwrap();
    let s0_spy = data1_latest.column("SPY")?.f64()?.get(0).unwrap();

    let x = process_options(df_opt, s0_nvda, "NVDA")?;
    Ok(x)
}
