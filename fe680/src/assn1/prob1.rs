//! This problem involves filling in a table of different rates based on input rates
//! which is then converted to other rates.
//!
//! We are interested in obtaining, for a list of input rates, the forward rate `f`, the discount
//! factor `d`, the zero curve `s`, and the par coupon `c`.
//!
//! We will use the bootstrapping process to obtain these values, starting from the smallest maturity.
//!

/// There are four different types of rates under consideration:
/// - The **overnight** rate, which is the interest rate that banks charge to lend each other
/// money overnight.
/// - The **cash** rate, which is the interest
enum InputRateType {
    Overnight,
    Cash,
    Forwards,
    Swaps,
}

struct InputRate {
    rate: f64,
    rate_type: InputRateType,
    /// Time to maturity in years
    maturity: u8,
}

/// Converts the forward rate to the discount factor
pub fn f_to_d(f: f64) -> f64 {
    1.0 / (1.0 + f)
}

/// Converts the zero rate to the forward rate using a non-arbitrage argument.
/// In order to do this, it requires both the zero rate and the forward rate for other maturities.
pub fn z_to_f(z: f64, fs: Vec<f64>) -> f64 {
    let n = fs.len();
    fs.iter()
        .fold((1.0 + z).powf(n as f64), |acc, f| acc / (1.0 + f))
}
