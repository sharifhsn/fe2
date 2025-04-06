use std::f64;

use itertools::izip;
use rand::prelude::*;
use rand_distr::StandardNormal;

use argmin::{
    core::{CostFunction, Executor},
    solver::neldermead::NelderMead,
};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionStyle {
    European,
    American,
}

#[derive(Debug, Clone, Copy)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, Copy)]
pub struct Zcb {
    L: f64,
    s: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Bond {
    pub L: f64, // principal
    pub c: f64, // coupon rate
    pub m: f64, // coupon frequency (e.g. 0.5 for semiannually)
    pub s: f64, // time to maturity
}

impl Bond {
    pub fn price(&self, r: f64) -> f64 {
        let mut price = 0.0;
        let num_coupons = (self.s / self.m).ceil() as i32;
        for t in 1..=num_coupons {
            let time = t as f64 * self.m;
            price += self.c * self.L * (-r * time).exp();
        }
        price += self.L * (-r * self.s).exp();
        price
    }

    pub fn port_zcb(&self) -> Vec<Zcb> {
        let mut zcbs = Vec::new();
        let num_coupons = (self.s / self.m).ceil() as i32;
        for t in 1..num_coupons {
            let time = t as f64 * self.m;
            zcbs.push(Zcb {
                L: self.c * self.L,
                s: time,
            });
        }
        zcbs.push(Zcb {
            L: self.L + self.c * self.L,
            s: self.s,
        });
        zcbs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vasicek {
    pub r0: f64,    // current interest rate
    pub a: f64,     // mean reversion rate
    pub b: f64,     // long-term mean
    pub sigma: f64, // volatility
}

impl Vasicek {
    pub fn dr(&self, r: f64, dt: f64) -> f64 {
        let epsilon: f64 = rand::rng().sample(StandardNormal);
        self.a * (self.b - r) * dt - 0.5 * self.sigma.powi(2) * dt.sqrt() * epsilon
    }

    fn B(&self, t: f64, T: f64) -> f64 {
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

#[derive(Debug, Clone)]
pub struct BondOption {
    pub bond: Bond,
    pub K: f64, // quoted strike price
    pub T: f64, // time to maturity
    pub option_style: OptionStyle,
    pub option_type: OptionType,
    pub vasicek: Vasicek,
}

impl BondOption {
    pub fn price(&self) -> f64 {
        // run the Jamshidian procedure to get rK
        let jamshidian = Jamshidian {
            bond_option: self.clone(),
        };
        let init_param = self.vasicek.r0;
        let solver = NelderMead::new(vec![
            vec![init_param, init_param + 0.1],
            vec![init_param - 0.1, init_param + 0.2],
        ]);
        let executor = Executor::new(jamshidian, solver);
        let result = executor.run().unwrap();
        let rK = result.state.best_param.unwrap()[0];

        // get the input zero coupon bonds and implied strikes
        let zcbs = self.bond.port_zcb();
        let zcbs = zcbs.iter().filter(|zcb| zcb.s > self.T).collect::<Vec<_>>();
        let Ks = zcbs
            .iter()
            .map(|zcb| zcb.L * self.vasicek.P(self.T, zcb.s, rK))
            .collect::<Vec<_>>();

        // calculate option price from sum of zcb prices
        let mut total = 0.0;
        for (zcb, K) in izip!(zcbs, Ks) {
            let Ps = self.vasicek.P0(zcb.s);
            let PT = self.vasicek.P0(self.T);

            let sigma_P = self.vasicek.sigma / self.vasicek.a
                * (1.0 - (-self.vasicek.a * (zcb.s - self.T)).exp())
                * ((1.0 - (-2.0 * self.vasicek.a * self.T).exp()) / (2.0 * self.vasicek.a)).sqrt();

            let h = ((zcb.L * Ps) / (PT * K)).ln() / sigma_P + sigma_P / 2.0;

            let normal = Normal::new(0.0, 1.0).unwrap();

            let price = match self.option_type {
                OptionType::Call => zcb.L * Ps * normal.cdf(h) - K * PT * normal.cdf(h - sigma_P),
                OptionType::Put => K * PT * normal.cdf(-h + sigma_P) - zcb.L * Ps * normal.cdf(-h),
            };
            total += price;
        }
        total
    }
}

#[derive(Debug)]
pub struct Jamshidian {
    bond_option: BondOption,
}

impl CostFunction for Jamshidian {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, rK: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        let rK = rK[0];
        let zcbs = self.bond_option.bond.port_zcb();
        // we don't consider the coupon payments before option maturity when pricing the option
        let zcbs_in_option = zcbs
            .iter()
            .filter(|zcb| zcb.s > self.bond_option.T)
            .collect::<Vec<_>>();
        Ok((zcbs_in_option
            .iter()
            .map(|zcb| zcb.L * self.bond_option.vasicek.P(self.bond_option.T, zcb.s, rK))
            .sum::<f64>()
            - self.bond_option.K)
            .powi(2))
    }
}

pub fn a() {
    let bond = Bond {
        L: 100.0,
        c: 0.05,
        m: 0.5,
        s: 3.0,
    };

    let vasicek = Vasicek {
        r0: 0.06,
        a: 0.05,
        b: 0.08,
        sigma: 0.015,
    };

    let bond_option = BondOption {
        bond,
        K: 99.0,
        T: 2.1,
        option_style: OptionStyle::European,
        option_type: OptionType::Call,
        vasicek,
    };

    println!("{}", bond_option.price());
}
