use statrs::distribution::{ContinuousCDF, Normal};

enum OptionType {
    Call,
    Put
}

fn black_scholes(
    option_type: OptionType,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    sig: f64
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let d1 = (S.ln() - K.ln() + (r + sig.powi(2) / 2.0) * T) / (sig * T.sqrt());
    let d2 = d1 - sig * T.sqrt();
    
    match option_type {
        OptionType::Call => S * normal.cdf(d1) - K * (-r * T).exp() * normal.cdf(d2),
        OptionType::Put => K * (-r * T).exp() * normal.cdf(-d2) - S * normal.cdf(-d1),
    }
}

pub fn a() {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let S: f64 = 100.0; // Current stock price
    let K: f64 = 100.0; // Strike price
    let T: f64 = 1.0; // Time to expiration in years
    let r: f64 = 0.05; // Risk-free interest rate
    let sig: f64 = 0.2; // Volatility of the underlying stock
    let Q: f64 = 100.0; // Binary payoff

    let d1 = |K: f64| (S.ln() - K.ln() + (r + sig.powi(2) / 2.0) * T) / (sig * T.sqrt());
    let d2 = |K: f64| d1(K) - sig * T.sqrt();

    let Nd1 = |K| normal.cdf(d1(K));
    let Nd2 = |K| normal.cdf(d2(K));

    let delta: f64 = 0.1;
    let Kd = K + delta;

    let bisection = |delta: f64| S * (Nd1(K) - Nd1(Kd)) + Kd * Nd2(Kd) - K * Nd2(K) - Q * Nd1(K);
    let mut a = 1e-4;
    let mut b = 4.0;
    let ε = 1e-6;
    let max_iter = 1000;

    if (bisection(a) * bisection(b)).is_sign_positive() {
        eprintln!("f(a) and f(b) must have opposite signs");
        panic!("PANIC!");
    }

    let mut mid = (a + b) / 2.0;
    for _ in 0..max_iter {
        mid = (a + b) / 2.0;

        if bisection(mid).abs() < ε {
            println!("Root found: {}", mid);
        }

        if (bisection(mid) * bisection(a)).is_sign_negative() {
            b = mid;
        } else {
            a = mid;
        }
    }

    
    eprintln!("Maximum iterations reached without finding the root.");
    panic!("PANIC!");

}