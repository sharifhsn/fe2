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
    let calls_lf = calls
        .lazy()
        .with_column((col("expiration") / lit(60 * 60 * 24)).cast(DataType::Date))
        .with_column((col("last_trade_date") / lit(60 * 60 * 24)).cast(DataType::Date))
        .with_column(lit("call").alias("option_type"));
    let puts_lf = puts
        .lazy()
        .with_column((col("expiration") / lit(60 * 60 * 24)).cast(DataType::Date))
        .with_column((col("last_trade_date") / lit(60 * 60 * 24)).cast(DataType::Date))
        .with_column(lit("put").alias("option_type"));
    let lf = concat([calls_lf, puts_lf], Default::default())?;
    println!("{:?}", lf.collect().unwrap());
    Ok(())
}
