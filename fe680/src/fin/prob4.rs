use statrs::distribution::{Binomial, Discrete, DiscreteCDF};

pub struct CreditDefaultObligation {
    /// Notional of each credit in CDO
    pub L: f64,
    /// Number of credits in CDO
    pub Nc: usize,
    /// Recovery rate
    pub R: f64,
    /// Probability of default
    pub pd: f64,
    /// Senior tranche proportion (in credits)
    pub s: f64,
    /// Mezzanine tranche proportion (in credits)
    pub m: f64,
    /// Equity tranche proportion (in credits)
    pub e: f64,
}

impl CreditDefaultObligation {
    /// Constructor to initialize the CDO
    pub fn new(L: f64, Nc: usize, R: f64, pd: f64, s: f64, m: f64, e: f64) -> Self {
        CreditDefaultObligation {
            L,
            Nc,
            R,
            pd,
            s,
            m,
            e,
        }
    }

    /// probability of wipeout given number of defaults required
    pub fn p(&self, x: f64) -> f64 {
        let binomial = Binomial::new(self.pd, self.Nc as u64).unwrap();
        let x = (x / (1.0 - self.R)).floor();
        1.0 - binomial.cdf(x as u64) // CDF of Binomial distribution
    }

    /// price of tranche given number of credits x
    /// and number of prior defaults in waterfall y
    pub fn price(&self, x: f64, y: f64) -> f64 {
        let binomial = Binomial::new(self.pd, self.Nc as u64).unwrap();
        let mut total_loss = 0.0;
        for i in 0..=self.Nc {
            let loss = ((1.0 - self.R) * i as f64 - y).max(0.0).min(x);
            total_loss += loss * binomial.pmf(i as u64);
        }
        self.L * ((x - y) - total_loss)
    }

}

pub fn a() {
    let L = 1_000_000.0;
    let Nc = 100;
    let R = 0.3;
    let pd = 0.18;
    let s = 80.0;
    let m = 12.0;
    let e = 8.0;
    let cdo: CreditDefaultObligation = CreditDefaultObligation::new(L, Nc, R, pd, s, m, e);
    let ps = cdo.p(cdo.s + cdo.m + cdo.e);
    let pm = cdo.p(cdo.m + cdo.e);
    let pe = cdo.p(cdo.e);
    println!("Probability of wipeout for senior tranche: {:.4}", ps);
    println!("Probability of wipeout for mezzanine tranche: {:.4}", pm);
    println!("Probability of wipeout for equity tranche: {:.4}", pe);
}

pub fn b() {
    let L = 1_000_000.0;
    let Nc = 100;
    let R = 0.3;
    let pd = 0.18;
    let s = 80.0;
    let m = 12.0;
    let e = 8.0;
    let cdo = CreditDefaultObligation::new(L, Nc, R, pd, s, m, e);
    let price_senior = cdo.price(cdo.s + cdo.m + cdo.e, cdo.m + cdo.e);
    let price_mezzanine = cdo.price(cdo.m + cdo.e, cdo.e);
    let price_equity = cdo.price(cdo.e, 0.0);
    println!("Price of senior tranche: {:.4}", price_senior);
    println!("Price of mezzanine tranche: {:.4}", price_mezzanine);
    println!("Price of equity tranche: {:.4}", price_equity);
}