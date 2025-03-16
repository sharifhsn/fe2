use anyhow::Result;
use yahoo_finance_api as yf;

use polars::prelude::*;

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

#[tokio::main]
pub async fn a() -> Result<()> {
    let provider = yf::YahooConnector::new()?;
    let resp = provider.get_latest_quotes("NVDA", "1d").await?;
    let response = provider.search_options("NVDA").await?;
    let option_chain = response.option_chain.result;
    println!("Underlying Symbol: {}", option_chain[0].underlying_symbol);
    println!("Expiration Dates: {:?}", option_chain[0].expiration_dates);
    println!("Strikes: {:?}", option_chain[0].strikes);
    println!("Has Mini Options: {}", option_chain[0].has_mini_options);
    //println!("options: {:?}", option_chain.result[0].options[0].calls[0]);
    let calls = struct_to_dataframe!(
        &option_chain[0].options[0].calls,
        [
            contract_symbol,
            strike,
            currency,
            last_price,
            change,
            percent_change,
            volume,
            open_interest,
            bid,
            ask,
            contract_size,
            expiration,
            last_trade_date,
            implied_volatility,
            in_the_money
        ]
    )?;
    let puts = struct_to_dataframe!(
        &option_chain[0].options[0].puts,
        [
            contract_symbol,
            strike,
            currency,
            last_price,
            change,
            percent_change,
            volume,
            open_interest,
            bid,
            ask,
            contract_size,
            expiration,
            last_trade_date,
            implied_volatility,
            in_the_money
        ]
    )?;
    let calls_df = calls
        .lazy()
        .with_column((col("expiration") / lit(60 * 60 * 24)).cast(DataType::Date))
        .with_column((col("last_trade_date") / lit(60 * 60 * 24)).cast(DataType::Date))
        .with_column(lit("call").alias("option_type"))
        .collect()?;
    let atm_price = calls_df
        .column("strike")?
        .f64()?
        .into_iter()
        .zip(calls_df.column("in_the_money")?.bool()?.into_iter())
        .filter_map(|(strike, in_the_money)| {
            if let (Some(strike), Some(in_the_money)) = (strike, in_the_money) {
                Some((strike, in_the_money))
            } else {
                None
            }
        })
        .find(|&(_, in_the_money)| !in_the_money)
        .map(|(strike, _)| strike)
        .unwrap_or(0.0);

    let calls_df = calls_df
        .lazy()
        .with_column(
            (col("strike") - lit(atm_price))
                .abs()
                .alias("distance_to_atm"),
        )
        .sort(["distance_to_atm"], Default::default())
        .collect()?
        .head(Some(30))
        .sort(["strike"], Default::default())?;
    println!(
        "{:?}",
        calls_df.select(["strike", "implied_volatility", "expiration"])
    );
    println!("{:#?}", option_chain[0].options);
    use plotters::prelude::*;

    let root_area = BitMapBackend::new("volatility_smile.png", (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Volatility Smile", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            calls_df.column("strike")?.f64()?.min().unwrap()
                ..calls_df.column("strike")?.f64()?.max().unwrap(),
            calls_df.column("implied_volatility")?.f64()?.min().unwrap()
                ..calls_df.column("implied_volatility")?.f64()?.max().unwrap(),
        )?;

    chart.configure_mesh().draw()?;
    let strikes: Vec<f64> = calls_df
        .column("strike")?
        .f64()?
        .to_vec_null_aware()
        .left()
        .unwrap();
    let implied_volatilities: Vec<f64> = calls_df
        .column("implied_volatility")?
        .f64()?
        .to_vec_null_aware()
        .left()
        .unwrap();

    chart.draw_series(LineSeries::new(
        strikes
            .iter()
            .zip(implied_volatilities.iter())
            .map(|(&x, &y)| (x, y)),
        &RED,
    ))?;

    root_area.present()?;
    // let puts_lf = puts
    //     .lazy()
    //     .with_column((col("expiration") / lit(60 * 60 * 24)).cast(DataType::Date))
    //     .with_column((col("last_trade_date") / lit(60 * 60 * 24)).cast(DataType::Date))
    //     .with_column(lit("put").alias("option_type"));
    // let lf = concat([calls_lf, puts_lf], Default::default())?;
    // println!("{:?}", lf.collect().unwrap());
    Ok(())
}
