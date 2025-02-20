//! This problem involves filling in a table of different rates based on input rates
//! which is then converted to other rates.
//!
//! We are interested in obtaining, for a list of input rates, the forward rate `f`, the discount
//! factor `d`, the zero curve `s`, and the par coupon `c`.
//!
//! We will use the bootstrapping process to obtain these values, starting from the smallest maturity.
//!
//! There are four different types of rates under consideration:
//! - The **overnight** rate, which is the interest rate that banks charge to lend each other money overnight.
//! - The **cash** rate, which is the same rate of lending for short periods of time.

use std::f64::consts::E;

use polars::df;
use polars::prelude::*;

pub fn bootstrap_forward(mut df: DataFrame) -> PolarsResult<DataFrame> {
    let mut bootstrapped_forward_curve = match df.column("forward_curve") {
        Ok(col) => col.f64()?.to_vec(),
        // forward curve does not exist, initialize with first zero curve value
        Err(_) => {
            let mut b = vec![None; df.height()];
            b[0] = Some(df.column("zero_curve")?.f64()?.get(0).unwrap());
            b
        }
    };

    // initialize with value with zero curve, since zero = forward for maturity = 1
    bootstrapped_forward_curve[0] = Some(df.column("zero_curve")?.f64()?.get(0).unwrap());
    // bootstrap from start of null
    let start_idx = bootstrapped_forward_curve
        .iter()
        .position(|&x| x.is_none())
        .unwrap();
    for i in start_idx..df.height() {
        let zero_val_opt = df.column("zero_curve")?.f64()?.get(i);

        bootstrapped_forward_curve[i] = if let Some(zero_val) = zero_val_opt {
            let maturity_val = df.column("maturity")?.f64()?.get(i).unwrap();
            let previous_forward_vals = &bootstrapped_forward_curve[..i];
            let prod: f64 = previous_forward_vals
                .iter()
                .map(|x| 1.0 / (1.0 + x.unwrap_or(0.0)))
                .product();
            Some((zero_val + 1.0).powf(maturity_val) * prod - 1.0)
        } else {
            None
        };
    }
    let bootstrapped_series = Column::new(
        "bootstrapped_forward_curve".into(),
        bootstrapped_forward_curve,
    );
    let forward_series = df.column("forward_curve")?;
    let mask = forward_series.is_not_null();
    let updated_forward_series = forward_series.zip_with(&mask, &bootstrapped_series)?;
    df.replace(
        "forward_curve",
        updated_forward_series.as_series().unwrap().clone(),
    )?;
    Ok(df)
}

fn bootstrap_discount(mut df: DataFrame) -> PolarsResult<DataFrame> {
    let mut bootstrapped_discount_curve = df.column("discount_curve")?.f64()?.to_vec();
    let start_idx = bootstrapped_discount_curve
        .iter()
        .position(|&x| x.is_none())
        .unwrap();
    for i in start_idx..bootstrapped_discount_curve.len() {
        let par_val_opt = df.column("par_curve")?.f64()?.get(i);
        bootstrapped_discount_curve[i] = if let Some(par_val) = par_val_opt {
            let previous_discount_vals = &bootstrapped_discount_curve[..i];
            let sum: f64 = previous_discount_vals
                .iter()
                .map(|x| x.unwrap_or(0.0))
                .sum();
            Some((1.0 - par_val * sum) / (1.0 + par_val))
        } else {
            None
        };
    }
    df.replace(
        "discount_curve",
        Series::new("discount_curve".into(), bootstrapped_discount_curve),
    )?;
    Ok(df)
}

const DY: f64 = 0.001;
const BP: f64 = 0.0001;

pub fn dv01(lf: &LazyFrame) -> PolarsResult<f64> {
    let df = lf
        .clone()
        .with_column((col("discount_curve_shifted") / lit(1.0 + BP)).alias("discount_curve_up"))
        .with_column((col("discount_curve_shifted") * lit(1.0 + BP)).alias("discount_curve_down"))
        .with_column(
            (col("discount_curve_up") * col("bond_cash_flow")).alias("pv_bond_cash_flow_up"),
        )
        .with_column(
            (col("discount_curve_down") * col("bond_cash_flow")).alias("pv_bond_cash_flow_down"),
        )
        .collect()?;
    Ok((df.column("pv_bond_cash_flow_up")?.f64()?.sum().unwrap()
        - df.column("pv_bond_cash_flow_down")?.f64()?.sum().unwrap())
        / 2.0)
}

pub fn duration(lf: &LazyFrame) -> PolarsResult<f64> {
    let df = lf
        .clone()
        .with_column((col("discount_curve_shifted") / lit(1.0 + BP)).alias("discount_curve_up"))
        .with_column((col("discount_curve_shifted") * lit(1.0 + BP)).alias("discount_curve_down"))
        .with_column(
            (col("discount_curve_up") * col("bond_cash_flow")).alias("pv_bond_cash_flow_up"),
        )
        .with_column(
            (col("discount_curve_down") * col("bond_cash_flow")).alias("pv_bond_cash_flow_down"),
        )
        .collect()?;
    Ok((df.column("pv_bond_cash_flow_up")?.f64()?.sum().unwrap()
        - df.column("pv_bond_cash_flow_down")?.f64()?.sum().unwrap())
        / (2.0 * df.column("pv_bond_cash_flow")?.f64()?.sum().unwrap() * BP))
}

fn pv(lf: &LazyFrame) -> PolarsResult<f64> {
    Ok(lf
        .clone()
        .collect()?
        .column("pv_bond_cash_flow")?
        .f64()?
        .sum()
        .unwrap())
}

#[derive(Default, Debug)]
pub struct Scenario {
    pub desc: String,
    pub discount_curve: Vec<f64>,
    pub pv: f64,
    pub dv01: f64,
    pub duration: f64,
}

pub fn a() -> PolarsResult<DataFrame> {
    // Initialize table
    let df: DataFrame = df!(
        "rate_type" => ["cash", "cash", "forwards", "forwards", "forwards", "swaps", "swaps", "swaps", "swaps", "swaps"],
        "maturity" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "inputs" => [0.03180, 0.03222, 0.03261, 0.03290, 0.03345, 0.03405, 0.03442, 0.03350, 0.03300, 0.03541],
        "bond_cash_flow" => [3, 3, 3, 3, 3, 3, 3, 3, 103, 0],
    )?;

    // Fill in given values
    let lf = df
        .lazy()
        .with_column(
            when(col("rate_type").eq(lit("cash")))
                .then(col("inputs"))
                .otherwise(lit(NULL))
                .alias("zero_curve"),
        )
        .with_column(
            when(col("rate_type").eq(lit("forwards")))
                .then(col("inputs"))
                .otherwise(lit(NULL))
                .alias("forward_curve"),
        )
        .with_column(
            when(col("rate_type").eq(lit("swaps")))
                .then(col("inputs"))
                .otherwise(lit(NULL))
                .alias("par_curve"),
        );
    // Bootstrap forwards from zero curve
    let df = lf.collect()?;
    let df = bootstrap_forward(df)?;
    // Get discounts from forward curve
    let lf = df
        .lazy()
        .with_column(
            when(col("forward_curve").is_not_null())
                .then(lit(1.0) / (lit(1.0) + col("forward_curve")).cum_prod(false))
                .otherwise(lit(NULL))
                .alias("discount_curve"),
        )
        // Get par from discount curve
        .with_column(
            when(col("discount_curve").is_not_null())
                .then((lit(1.0) - col("discount_curve")) / (col("discount_curve").cum_sum(false)))
                .otherwise(col("par_curve"))
                .alias("par_curve"),
        );

    // Bootstrap discount from par curve
    let df = lf.collect()?;
    let df = bootstrap_discount(df)?;

    // Get zero curve from discount curve
    let lf = df.lazy().with_column(
        when(col("zero_curve").is_null())
            .then(col("discount_curve").pow(lit(-1.0) / col("maturity")) - lit(1.0))
            .otherwise(col("zero_curve"))
            .alias("zero_curve"),
    );

    // Continue bootstrapping forward curve with finished zero curve
    let df = lf.collect()?;
    let df = bootstrap_forward(df)?;
    println!("{:?}", df);

    // b)
    // Get present value of the bond cash flows
    let lf = df
        .lazy()
        .with_column((col("discount_curve") * col("bond_cash_flow")).alias("pv_bond_cash_flow"));
    // let lf = dv01(lf)?;
    let df = lf.collect()?;
    println!("{:?}", df);
    println!(
        "Bond Price: {}",
        df.column("pv_bond_cash_flow")?.f64()?.sum().unwrap()
    );

    // c)
    // add row index
    let df = df.with_row_index("row".into(), None)?;
    let scenarios = (0..df.height())
        .map(|i| {
            let scenario = df
                .clone()
                .lazy()
                // increment forward curve by 0.1% for each maturity
                .with_column(
                    when(col("row").eq(lit(i as u32)))
                        .then(col("forward_curve") + lit(0.001))
                        .otherwise(col("forward_curve"))
                        .alias("forward_curve_shifted"),
                )
                // new discount factors
                .with_column(
                    (lit(1.0) / (lit(1.0) + col("forward_curve_shifted")).cum_prod(false))
                        .alias("discount_curve_shifted"),
                )
                // new pv
                .with_column(
                    (col("discount_curve_shifted") * col("bond_cash_flow"))
                        .alias("pv_bond_cash_flow"),
                );

            let pv = df
                .column("pv_bond_cash_flow")
                .unwrap()
                .f64()
                .unwrap()
                .sum()
                .unwrap();
            // new dv01
            let dv01 = dv01(&scenario).unwrap();
            let duration = duration(&scenario).unwrap();
            let df_scenario = scenario.collect().unwrap();
            // new duration
            Scenario {
                desc: format!("incremented maturity is {}", i + 1),
                discount_curve: df_scenario
                    .column("discount_curve_shifted")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .to_vec()
                    .into_iter()
                    .map(|x| x.unwrap())
                    .collect(),
                pv,
                dv01,
                duration,
            }
        })
        .collect::<Vec<Scenario>>();
    //println!("{:#?}", scenarios);
    // original scenario
    let df = df
        .lazy()
        .with_column(col("discount_curve").alias("discount_curve_shifted"))
        .collect()?;
    let scenario = Scenario {
        desc: String::from("Original"),
        discount_curve: df
            .column("discount_curve")
            .unwrap()
            .f64()
            .unwrap()
            .to_vec()
            .into_iter()
            .map(|x| x.unwrap())
            .collect(),
        pv: pv(&df.clone().lazy())?,
        dv01: dv01(&df.clone().lazy())?,
        duration: duration(&df.clone().lazy())?,
    };
    println!("original scenario: {:?}", scenario);
    // the price remains the same
    println!(
        "all_pvs {:?}",
        scenarios.iter().map(|s| s.pv).collect::<Vec<_>>()
    );
    // the dv01 is lowest when the first maturity is shifted, and gets closest to original
    // when last maturity is shifted
    println!(
        "all_dv01s {:?}",
        scenarios.iter().map(|s| s.dv01).collect::<Vec<_>>()
    );
    // the duration is basically identical
    println!(
        "all_durations {:?}",
        scenarios.iter().map(|s| s.duration).collect::<Vec<_>>()
    );

    // d)
    let df_shifted_up = df
        .clone()
        .lazy()
        .with_column((col("forward_curve") + lit(0.001)).alias("forward_curve_shifted"))
        .with_column(
            (lit(1.0) / (lit(1.0) + col("forward_curve_shifted")))
                .cum_prod(false)
                .alias("discount_curve_shifted"),
        )
        .with_column(
            (col("discount_curve_shifted") * col("bond_cash_flow")).alias("pv_bond_cash_flow"),
        )
        .collect()?;
    let df_shifted_down = df
        .clone()
        .lazy()
        .with_column((col("forward_curve") - lit(0.001)).alias("forward_curve_shifted"))
        .with_column(
            (lit(1.0) / (lit(1.0) + col("forward_curve_shifted")))
                .cum_prod(false)
                .alias("discount_curve_shifted"),
        )
        .with_column(
            (col("discount_curve_shifted") * col("bond_cash_flow")).alias("pv_bond_cash_flow"),
        )
        .collect()?;
    let pv_up = pv(&df_shifted_up.lazy())?;
    let pv_down = pv(&df_shifted_down.lazy())?;
    let pv = pv(&df.clone().lazy())?;
    println!("{}", pv_up);
    let convexity = (pv_up + pv_down - 2.0 * pv) / (pv * (0.001f64).powi(2));
    println!("Convexity: {}", convexity);

    // e)
    // I will use linear interpolation on the discount curve.
    let discount_curve = df
        .column("discount_curve")?
        .f64()?
        .to_vec()
        .iter()
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let discount30 =
        discount_curve[1] + (discount_curve[2] - discount_curve[1]) * (2.5 - 2.0) / (3.0 - 2.0);
    let forward30 = pv / discount30;
    println!("Forward (30 months): {}", forward30);
    Ok(df)
}
