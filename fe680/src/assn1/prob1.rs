//! This problem involves filling in a table of different rates based on input rates
//! which is then converted to other rates.
//!
//! We are interested in obtaining, for a list of input rates, the forward rate `f`, the discount
//! factor `d`, the zero curve `s`, and the par coupon `c`.
//!
//! We will use the bootstrapping process to obtain these values, starting from the smallest maturity.
//!
//!
//!
use polars::df;
use polars::prelude::*;

/// There are four different types of rates under consideration:
/// - The **overnight** rate, which is the interest rate that banks charge to lend each other money overnight.
/// - The **cash** rate, which is the same rate of lending for short periods of time.
enum InputRateType {
    Overnight,
    Cash,
    Forwards,
    Swaps,
}

// struct InputRate {
//     rate: f64,
//     rate_type: InputRateType,
//     /// Time to maturity in years
//     maturity: u8,
// }

/// Converts the forward rate to the discount factor
pub fn f_to_d(f: f64) -> f64 {
    1.0 / (1.0 + f)
}

fn z_to_d(t: f64, z: f64) -> f64 {
    1.0 / (1.0 + z).powf(t)
}

fn dusc(columns: &mut [Column]) -> PolarsResult<Option<Column>> {
    // Ensure we have exactly two columns.
    if columns.len() != 2 {
        return Err(PolarsError::ComputeError(
            "Expected exactly 2 columns".into(),
        ));
    }

    // Convert the columns to ChunkedArray<f64>.
    let maturity = columns[0].f64()?;
    let zero_curve = columns[1].f64()?;

    // Compute the discount curve by multiplying corresponding elements.
    // We iterate element-wise over the two ChunkedArrays.
    let result: Vec<Option<f64>> = maturity
        .into_iter()
        .zip(zero_curve)
        .map(|(t, z)| match (t, z) {
            (Some(t_val), Some(z_val)) => Some(z_to_d(t_val, z_val)),
            _ => None,
        })
        .collect();

    Ok(Some(Column::new("discount_curve".into(), result)))
}

/// Converts the zero rate to the forward rate using a non-arbitrage argument.
/// In order to do this, it requires both the zero rate and the forward rate for other maturities.
pub fn z_to_f(z: f64, fs: Series) -> f64 {
    let n = fs.len();
    fs.f64()
        .expect("Expected a float series")
        .iter()
        .fold((1.0 + z).powf(n as f64), |acc, f| {
            acc / (1.0 + f.unwrap_or_default())
        })
}

// Let's get the par coupon c_2

pub fn a() {
    // Initialize table
    let mut df = df!(
        "rate_type" => ["cash", "cash", "forwards", "forwards", "forwards", "swaps", "swaps", "swaps", "swaps", "swaps"],
        "maturity" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "inputs" => [0.03180, 0.03222, 0.03261, 0.03290, 0.03345, 0.03405, 0.03442, 0.03350, 0.03300, 0.03541],
        // "discount_curve" => [None::<f64>, None, None, None, None, None, None, None, None, None, None],
        // "zero_curve" => [None::<f64>, None, None, None, None, None, None, None, None, None, None],
        // "forward_curve" => [None::<f64>, None, None, None, None, None, None, None, None, None, None],
        // "par_curve" => [None::<f64>, None, None, None, None, None, None, None, None, None, None],
        "bond_cash_flow" => [3, 3, 3, 3, 3, 3, 3, 3, 103, 0],
    ).unwrap();
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
        )
        // get discount curve from zero curve
        .with_column(
            col("maturity")
                .map_many(
                    dusc,
                    &[col("zero_curve")],
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("discount_curve"),
        );
    println!("{:?}", lf.collect());
}
