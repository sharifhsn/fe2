const BP: f64 = 0.0001;

const RF: f64 = 0.04;

fn P(T: f64) -> f64 {
    (-RF * T).exp()
}

pub struct CreditDefaultSwap {
    pub T: Vec<f64>,
    pub R: f64,
    pub s: f64,
}

impl CreditDefaultSwap {
    pub fn new(T: Vec<f64>, R: f64, s: f64) -> Self {
        CreditDefaultSwap { T, R, s }
    }

    pub fn hazard_to_spread(&self, lambda: f64) -> f64 {
        let Q: Vec<f64> = self.T.iter().map(|t| (-lambda * t).exp()).collect();
        let mut D: Vec<f64> = vec![0.0];
        D.extend_from_slice(
            &(1..self.T.len())
                .map(|i| Q[i - 1] - Q[i])
                .collect::<Vec<_>>(),
        );

        // PV premium
        let pv_prem: f64 = (1..self.T.len()).map(|i| P(self.T[i]) * Q[i]).sum();

        // PV protection
        let pv_protection: f64 = (1..self.T.len())
            .map(|i| P((self.T[i - 1] + self.T[i]) / 2.0) * D[i] * (1.0 - self.R))
            .sum();

        // PV premium accrual
        let pv_accrual: f64 = (1..self.T.len())
            .map(|i| 0.5 * P((self.T[i - 1] + self.T[i]) / 2.0) * D[i])
            .sum();

        let s = pv_protection / (pv_prem + pv_accrual);
        self.s - s
    }

    fn bisect(&self, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<f64> {
        if (self.hazard_to_spread(a) * self.hazard_to_spread(b)).is_sign_positive() {
            eprintln!("f(a) and f(b) must have opposite signs");
            return None;
        }

        let mut mid;
        for _ in 0..max_iter {
            mid = (a + b) / 2.0;

            if self.hazard_to_spread(mid).abs() < tol {
                return Some(mid);
            }

            if (self.hazard_to_spread(mid) * self.hazard_to_spread(a)).is_sign_negative() {
                b = mid;
            } else {
                a = mid;
            }
        }
        eprintln!("Maximum iterations reached without finding the root.");
        None
    }
}


pub struct Company {
    /// spread for vanilla CDS
    sv: f64,
    /// spread for binary CDS
    sb: f64,
}

impl Company {
    pub fn new(sv: f64, sb: f64) -> Self {
        Self { sv, sb }
    }

    pub fn R(&self) -> f64 {
        1.0 - self.sv / self.sb
    }
}

pub fn a() {
    let sv = 145.0 * BP;
    let sb = 200.0 * BP;
    let company = Company::new(sv, sb);
    println!("Recovery rate: {}", company.R());

    let T = vec![1.0, 2.0, 3.0, 4.0,  5.0];
    let cds = CreditDefaultSwap::new(
        T, company.R(), sv
    );
    let maybe_hazard = cds.bisect(1e-3, 1.0, 1e-10, 1000);
    let hazard = maybe_hazard.unwrap();
    println!("Hazard: {:.4}", hazard);
}