use rand::prelude::*;
use rand_distr::StandardNormal;

const EPSILON: f64 = 1e-6;

pub enum Derivative {
    Numerical,
    Analytical,
}

#[derive(Debug, Clone, Copy)]
pub struct Vasicek {
    pub r0: f64,    // current interest rate
    pub a: f64,     // mean reversion rate
    pub b: f64,     // long-term mean
    pub sigma: f64, // volatility
}

#[derive(Debug, Clone, Copy)]
pub struct Zcb {
    L: f64,
    T: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Bond {
    pub L: f64, // principal
    pub c: f64, // coupon rate
    pub m: f64, // coupon frequency (e.g. 0.5 for semiannually)
    pub T: f64, // time to maturity
}

impl Bond {
    pub fn port_zcb(&self) -> Vec<Zcb> {
        let mut zcbs = Vec::new();
        let num_coupons = (self.T / self.m).ceil() as i32;
        for t in 1..num_coupons {
            let time = t as f64 * self.m;
            zcbs.push(Zcb {
                L: self.c * self.L,
                T: time,
            });
        }
        zcbs.push(Zcb {
            L: self.L + self.c * self.L,
            T: self.T,
        });
        zcbs
    }
    pub fn duration(&self, vasicek: Vasicek, d: Derivative) -> f64 {
        let price = self.price(vasicek);
        let num_coupons = (self.T / self.m).ceil() as i32;
        let mut total = 0.0;
        for t in 1..=num_coupons {
            let T = t as f64 * self.m;
            let derivative = match d {
                Derivative::Numerical => {
                    (vasicek.P(0.0, T, vasicek.r0 + EPSILON)
                        - vasicek.P(0.0, T, vasicek.r0 - EPSILON))
                        / (2.0 * EPSILON)
                }
                Derivative::Analytical => -vasicek.B(0.0, T) * vasicek.P0(T),
            };
            total += self.c * self.L * derivative;
        }
        let final_derivative = match d {
            Derivative::Numerical => {
                (vasicek.P(0.0, self.T, vasicek.r0 + EPSILON)
                    - vasicek.P(0.0, self.T, vasicek.r0 - EPSILON))
                    / (2.0 * EPSILON)
            }
            Derivative::Analytical => -vasicek.B(0.0, self.T) * vasicek.P0(self.T),
        };
        total += self.L * final_derivative;
        -total / price
    }
    pub fn price(&self, vasicek: Vasicek) -> f64 {
        let mut price = 0.0;
        let num_coupons = (self.T / self.m).ceil() as i32;
        for t in 1..=num_coupons {
            let time = t as f64 * self.m;
            price += vasicek.P0(time) * self.c * self.L;
        }
        price += self.L * vasicek.P0(self.T);
        price
    }
}

impl Vasicek {
    pub fn dr(&self, r: f64, dt: f64) -> f64 {
        let epsilon: f64 = rand::rng().sample(StandardNormal);
        self.a * (self.b - r) * dt - 0.5 * self.sigma.powi(2) * dt.sqrt() * epsilon
    }

    pub fn B(&self, t: f64, T: f64) -> f64 {
        (1.0 - (-self.a * (T - t)).exp()) / self.a
    }

    fn A(&self, t: f64, T: f64) -> f64 {
        ((self.B(t, T) - (T - t)) * (self.a.powi(2) * self.b - 0.5 * self.sigma.powi(2))
            / self.a.powi(2)
            - self.sigma.powi(2) * self.B(t, T).powi(2) / (4.0 * self.a))
            .exp()
    }

    pub fn P(&self, t: f64, T: f64, r_t: f64) -> f64 {
        self.A(t, T) * (-self.B(t, T) * r_t).exp()
    }

    pub fn P0(&self, T: f64) -> f64 {
        self.P(0.0, T, self.r0)
    }
}

pub fn a() {
    let bond = Bond {
        L: 100.0,
        c: 0.015,
        m: 0.5,
        T: 2.0,
    };
    let mut vasicek = Vasicek {
        r0: 0.01,
        a: 0.13,
        b: 0.012,
        sigma: 0.01,
    };
    println!("Price: {:.2}", bond.price(vasicek));
    println!(
        "Duration: {:.2}",
        bond.duration(vasicek, Derivative::Analytical)
    );
    println!(
        "Numerical Duration: {:.2}",
        bond.duration(vasicek, Derivative::Numerical)
    );

    let price_before = bond.price(vasicek);
    let r0_orig = vasicek.r0;
    vasicek.r0 += 0.0005; // Small change in interest rate
    let price_after = bond.price(vasicek);
    let predicted_change =
        -bond.duration(vasicek, Derivative::Analytical) * (vasicek.r0 - r0_orig) * price_before;
    let actual_change = price_after - price_before;

    println!("Price before: {:.2}", price_before);
    println!("Price after: {:.2}", price_after);
    println!("Predicted change: {:.2}", predicted_change);
    println!("Actual change: {:.2}", actual_change);
}
