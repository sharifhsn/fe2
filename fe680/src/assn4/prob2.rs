const RF: f64 = 0.04; // continuously compounded
const R: f64 = 0.3; // recovery rate

fn P(T: f64) -> f64 {
    (-RF * T).exp()
}

#[derive(Clone, Copy)]
pub enum CDSPayoff {
    Regular,
    Binary,
}

pub struct CreditDefaultSwap {
    pub T: Vec<f64>,
    pub D: Vec<f64>,
    pub payoff_type: CDSPayoff,
}

impl CreditDefaultSwap {
    pub fn new(T: Vec<f64>, D: Vec<f64>, payoff_type: CDSPayoff) -> Self {
        CreditDefaultSwap { T, D, payoff_type }
    }

    pub fn spread(&self) -> f64 {
        let mut Q: Vec<f64> = vec![1.0];
        for i in 1..self.T.len() {
            Q.push(Q[i - 1] - self.D[i]);
        }

        // PV premium
        let pv_prem: f64 = (1..self.T.len()).map(|i| P(self.T[i]) * Q[i]).sum();

        // PV protection
        let pv_protection: f64 = match self.payoff_type {
            CDSPayoff::Regular => (1..self.T.len())
                .map(|i| P((self.T[i - 1] + self.T[i]) / 2.0) * self.D[i] * (1.0 - R))
                .sum(),
            CDSPayoff::Binary => (1..self.T.len())
                .map(|i| P((self.T[i - 1] + self.T[i]) / 2.0) * self.D[i])
                .sum(),
        };

        // PV premium accrual
        let pv_accrual: f64 = (1..self.T.len())
            .map(|i| 0.5 * P((self.T[i - 1] + self.T[i]) / 2.0) * self.D[i])
            .sum();

        pv_protection / (pv_prem + pv_accrual)
    }
}

pub fn a() {
    let T = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let D = vec![0.0, 0.012, 0.012, 0.015, 0.015];

    // let T = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    // let D = vec![0.0, 0.0198, 0.0194, 0.0190, 0.0186, 0.0183];

    let cds = CreditDefaultSwap::new(T.clone(), D.clone(), CDSPayoff::Regular);
    let spread = cds.spread();
    println!("Spread: {}", spread);
    let bcds = CreditDefaultSwap::new(T, D, CDSPayoff::Binary);
    let bspread = bcds.spread();
    println!("Binary spread: {}", bspread);
}
