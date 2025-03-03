use itertools::Itertools;
use nalgebra::DMatrix;
use plotters::prelude::*;
use polars::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use std::{f64::consts::E, fs::File, path::PathBuf};

#[derive(Clone, Copy, Debug)]
enum OptionType {
    Call,
    Put,
}

#[derive(Clone, Copy, Debug)]
enum OptionStyle {
    European,
    American,
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
const R: f64 = 0.0433;

/// Represents an option priced using the Trigeorgis additive binomial tree.
#[derive(Debug)]
struct TrigeorgisBinomialTreeOption {
    s0: f64,                   // Initial stock price
    k: f64,                    // Strike price
    t: f64,                    // Time to maturity (years)
    r: f64,                    // Risk-free rate
    sigma: f64,                // Volatility
    steps: usize,              // Number of tree steps
    option_type: OptionType,   // Call or Put
    option_style: OptionStyle, // American or European
}

impl TrigeorgisBinomialTreeOption {
    /// Constructs a new option.
    fn new(
        s0: f64,
        k: f64,
        t: f64,
        r: f64,
        sigma: f64,
        steps: usize,
        option_type: OptionType,
        option_style: OptionStyle,
    ) -> Self {
        Self {
            s0,
            k,
            t,
            r,
            sigma,
            steps,
            option_type,
            option_style,
        }
    }

    /// Computes the option price using the Trigeorgis binomial tree.
    fn price(&self) -> f64 {
        let dt = self.t / self.steps as f64; // Time step
        let drift = self.r - 0.5 * self.sigma.powi(2); // Adjusted drift
        let dx = ((drift.powi(2) * dt.powi(2)) + (self.sigma.powi(2) * dt)).sqrt(); // Log return step size
        let pu = 0.5 + 0.5 * (drift * dt / dx); // Risk-neutral up probability
        let pd = 1.0 - pu; // Down probability
        let discount = (-self.r * dt).exp(); // Discount factor

        // Construct tree for log prices (X_t)
        let mut log_tree = DMatrix::zeros(self.steps + 1, self.steps + 1);
        for j in 0..=self.steps {
            for i in 0..=j {
                log_tree[(i, j)] = (self.s0.ln()) + (i as f64 * dx) + ((j - i) as f64 * (-dx));
            }
        }

        // Convert back to price domain
        let stock_tree = log_tree.map(|x| E.powf(x));

        // Compute option values at terminal nodes
        let mut option_tree = DMatrix::zeros(self.steps + 1, self.steps + 1);
        for i in 0..=self.steps {
            let intrinsic_value = match self.option_type {
                OptionType::Call => f64::max(stock_tree[(i, self.steps)] - self.k, 0.0),
                OptionType::Put => f64::max(self.k - stock_tree[(i, self.steps)], 0.0),
            };
            option_tree[(i, self.steps)] = intrinsic_value;
        }

        // Parallelized Backward Induction
        for j in (0..self.steps).rev() {
            for i in 0..=j {
                let continuation_value =
                    discount * (pu * option_tree[(i + 1, j + 1)] + pd * option_tree[(i, j + 1)]);

                option_tree[(i, j)] = match self.option_style {
                    OptionStyle::American => {
                        let intrinsic_value = match self.option_type {
                            OptionType::Call => f64::max(stock_tree[(i, j)] - self.k, 0.0),
                            OptionType::Put => f64::max(self.k - stock_tree[(i, j)], 0.0),
                        };
                        f64::max(continuation_value, intrinsic_value)
                    }
                    OptionStyle::European => continuation_value,
                };
            }
        }

        option_tree[(0, 0)] // Option price at time t=0
    }
}

fn compute_option_prices(
    df_opt: &mut DataFrame,
    s0: f64,
    r: f64,
    steps: usize,
) -> PolarsResult<()> {
    // Extract necessary columns
    let strikes = df_opt.column("strike")?.f64()?.to_vec();
    let ivs = df_opt.column("impliedVolatility")?.f64()?.to_vec();
    let maturities = df_opt.column("timeToMaturity")?.duration()?.to_vec();
    let option_types: Vec<Option<&str>> = df_opt.column("optionType")?.str()?.into_iter().collect();

    // Compute option prices in parallel
    let results: Vec<(f64, f64, f64)> = (strikes, ivs, maturities, option_types)
        .into_par_iter()
        .map(|(strike, iv, maturity, opt_type)| {
            if let (Some(k), Some(sigma), Some(expiry), Some(opt_str)) =
                (strike, iv, maturity, opt_type)
            {
                let τ = ((expiry / 86_400_000) as f64) / 365.0;
                let option_type = if opt_str == "call" {
                    OptionType::Call
                } else {
                    OptionType::Put
                };

                // Compute prices using different methods
                let price_bs = black_scholes(option_type, s0, sigma, τ, k, r);
                let price_tree_eu = TrigeorgisBinomialTreeOption::new(
                    s0,
                    k,
                    τ,
                    r,
                    sigma,
                    steps,
                    option_type,
                    OptionStyle::European,
                )
                .price();
                let price_tree_am = TrigeorgisBinomialTreeOption::new(
                    s0,
                    k,
                    τ,
                    r,
                    sigma,
                    steps,
                    option_type,
                    OptionStyle::American,
                )
                .price();

                (price_tree_am, price_tree_eu, price_bs)
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            }
        })
        .collect();

    let (american_prices, european_prices, black_scholes_prices): (Vec<f64>, Vec<f64>, Vec<f64>) =
        results.into_iter().multiunzip();
    // Add computed prices as new columns
    df_opt.with_column(Column::new("american_price".into(), american_prices))?;
    df_opt.with_column(Column::new("european_price".into(), european_prices))?;
    df_opt.with_column(Column::new(
        "black_scholes_price".into(),
        black_scholes_prices,
    ))?;

    Ok(())
}

pub fn a() {
    let call_option = TrigeorgisBinomialTreeOption::new(
        100.0,
        100.0,
        1.0,
        0.05,
        0.2,
        20000,
        OptionType::Call,
        OptionStyle::European,
    );

    println!("European Call Price: {:.4}", call_option.price());
    println!(
        "European Call Price (Black-Scholes): {:.4}",
        black_scholes(OptionType::Call, 100.0, 0.2, 1.0, 100.0, 0.05)
    );
}

pub fn b() -> PolarsResult<DataFrame> {
    let df_hist = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from("historical_data.csv")))?
        .finish()?;
    //println!("{:?}", df_hist);

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
            .head(Some(20))])
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

    println!("{s0_nvda}");
    compute_option_prices(&mut df_opt, s0_nvda, R, 500)?;

    let df_opt = df_opt
        .lazy()
        .with_column((col("american_price") - (col("bid") + col("ask")) / lit(2.0)).alias("diff"))
        .select([
            col("strike"),
            col("lastPrice"),
            col("bid"),
            col("ask"),
            col("impliedVolatility"),
            col("optionType"),
            col("expirationDate"),
            col("american_price"),
            col("european_price"),
            col("black_scholes_price"),
            col("diff"),
        ])
        .collect()?;

    // let root = BitMapBackend::new("option_prices.png", (1280, 960)).into_drawing_area();
    // root.fill(&WHITE).unwrap();
    println!("#J*#(J");
    let binding = df_opt.column("expirationDate")?.date()?.unique()?;
    let expiration_dates: Vec<_> = binding.into_iter().map(|x| x.unwrap()).collect();

    for expiration_date in expiration_dates.into_iter() {
        let df_filtered = df_opt
            .clone()
            .lazy()
            .filter(col("expirationDate").eq(lit(expiration_date)))
            .collect()?;

        let date = chrono::DateTime::from_timestamp((expiration_date * 86400) as i64, 0)
            .unwrap()
            .date_naive();

        for option_type in &["call", "put"] {
            let df_filtered_type = df_filtered
                .clone()
                .lazy()
                .filter(col("optionType").eq(lit(*option_type)))
                .collect()?;

            let s = format!("option_prices_{}_{}.png", date, option_type);
            File::create(&s).unwrap();
            let root = BitMapBackend::new(&s, (1280, 960)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let strikes: Vec<f64> = df_filtered_type
                .column("strike")?
                .f64()?
                .to_vec_null_aware()
                .left()
                .unwrap();
            let last_prices: Vec<f64> = df_filtered_type
                .column("lastPrice")?
                .f64()?
                .to_vec_null_aware()
                .left()
                .unwrap();
            let american_prices: Vec<f64> = df_filtered_type
                .column("american_price")?
                .f64()?
                .to_vec_null_aware()
                .left()
                .unwrap();
            let european_prices: Vec<f64> = df_filtered_type
                .column("european_price")?
                .f64()?
                .to_vec_null_aware()
                .left()
                .unwrap();
            let black_scholes_prices: Vec<f64> = df_filtered_type
                .column("black_scholes_price")?
                .f64()?
                .to_vec_null_aware()
                .left()
                .unwrap();
            let bids: Vec<f64> = df_filtered_type
                .column("bid")?
                .f64()?
                .to_vec_null_aware()
                .left()
                .unwrap();
            let asks: Vec<f64> = df_filtered_type
                .column("ask")?
                .f64()?
                .to_vec_null_aware()
                .left()
                .unwrap();

            let mut chart = ChartBuilder::on(&root)
                .caption(
                    format!(
                        "Option Prices for Expiration Date: {} ({})",
                        date, option_type
                    ),
                    ("sans-serif", 20),
                )
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(
                    *strikes
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                        ..*strikes
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                    *last_prices
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                        ..*last_prices
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap(),
                )
                .unwrap();

            chart.configure_mesh().draw().unwrap();

            chart
                .draw_series(LineSeries::new(
                    strikes
                        .iter()
                        .zip(last_prices.iter())
                        .map(|(&x, &y)| (x, y)),
                    &RED,
                ))
                .unwrap()
                .label("Last Price")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

            chart
                .draw_series(LineSeries::new(
                    strikes
                        .iter()
                        .zip(american_prices.iter())
                        .map(|(&x, &y)| (x, y)),
                    &BLUE,
                ))
                .unwrap()
                .label("American Price")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

            chart
                .draw_series(LineSeries::new(
                    strikes
                        .iter()
                        .zip(european_prices.iter())
                        .map(|(&x, &y)| (x, y)),
                    &GREEN,
                ))
                .unwrap()
                .label("European Price")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

            chart
                .draw_series(LineSeries::new(
                    strikes
                        .iter()
                        .zip(black_scholes_prices.iter())
                        .map(|(&x, &y)| (x, y)),
                    &BLACK,
                ))
                .unwrap()
                .label("Black-Scholes Price")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

            chart
                .draw_series(LineSeries::new(
                    strikes.iter().zip(bids.iter()).map(|(&x, &y)| (x, y)),
                    &MAGENTA,
                ))
                .unwrap()
                .label("Bid")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA));

            chart
                .draw_series(LineSeries::new(
                    strikes.iter().zip(asks.iter()).map(|(&x, &y)| (x, y)),
                    &CYAN,
                ))
                .unwrap()
                .label("Ask")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], CYAN));

            chart
                .configure_series_labels()
                .border_style(BLACK)
                .draw()
                .unwrap();

            root.present().unwrap();
        }
    }
    // println!("{df_opt}");
    // For calls, the American price is essentially identical to the European price, confirming
    // the existing theoretical framework.
    // This is also true for the short-dated put.
    // For the long-dated puts, the American price is a little worse at approximating the true price.

    let f = File::create("options.csv").unwrap();
    let mut c = CsvWriter::new(f);
    c.finish(&mut df_opt.clone()).unwrap();
    Ok(data2_latest)
}

pub fn d() {
    let s0 = 138.0;
    let k = 133.0;
    let t = 70.0 / 365.0;
    let sigma = 0.543949873046875;

    let ns: Vec<u32> = vec![10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400];
    let v: Vec<_> = ns
        .clone()
        .into_par_iter()
        .map(|n| {
            let call_option = TrigeorgisBinomialTreeOption::new(
                s0,
                k,
                t,
                R,
                sigma,
                n as usize,
                OptionType::Put,
                OptionStyle::European,
            );
            (black_scholes(OptionType::Call, s0, sigma, t, k, R) - call_option.price()).abs()
        })
        .collect();
    let df = df!(
        "N" => ns.clone(),
        "error" => v.clone(),
    )
    .unwrap();
    println!("{df}");

    // Create a plot using plotters
    let root = BitMapBackend::new("error_plot.png", (1280, 960)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Error vs Number of Steps", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            *ns.iter().min().unwrap()..*ns.iter().max().unwrap(),
            *v.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                ..*v.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        )
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            ns.iter().zip(v.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))
        .unwrap()
        .label("Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    // println!("European Call Price: {:.4}", call_option.price());
    // println!(
    //     "European Call Price (Black-Scholes): {:.4}",
    //     black_scholes(OptionType::Call, 100.0, 0.2, 1.0, 100.0, 0.05)
    // );
}
