use nalgebra::{dmatrix, dvector, DMatrix, DVector};

struct InvestmentUniverse {
    cov: DMatrix<f64>,
    inv_cov: DMatrix<f64>,
    ones: DVector<f64>,
    rho: DVector<f64>,
    rho0: f64,
}

impl InvestmentUniverse {
    pub fn new(cov: DMatrix<f64>, rho: DVector<f64>, rho0: f64) -> Self {
        InvestmentUniverse {
            inv_cov: cov
                .clone()
                .try_inverse()
                .expect("Covariance matrix is not invertible"),
            ones: DVector::from_element(rho.len(), 1.0),
            cov,
            rho,
            rho0,
        }
    }
    pub fn gmv(&self) -> DVector<f64> {
        let w_min_var = &self.inv_cov * &self.ones;
        let v = (self.ones.transpose() * &self.inv_cov * &self.ones)[(0, 0)];
        w_min_var / v
    }
    pub fn min_var_target(&self, target_return: f64) -> DVector<f64> {
        let a = (self.rho.transpose() * &self.inv_cov * &self.rho)[(0, 0)];
        let b = (self.rho.transpose() * &self.inv_cov * &self.ones)[(0, 0)];
        let c = (self.ones.transpose() * &self.inv_cov * &self.ones)[(0, 0)];

        let lambda = -2.0 * (c * target_return - b) / (a * c - b.powi(2));
        let gamma = -2.0 * (a - b * target_return) / (a * c - b.powi(2));

        -0.5 * &self.inv_cov * (lambda * &self.rho + gamma * &self.ones)
    }
    pub fn two_fund_theorem(&self, target_return: f64) -> DVector<f64> {
        let rho_1 = self.rho[0];
        let rho_avg_23 = (self.rho[1] + self.rho[2]) / 2.0;

        let w_min_var_1 = self.min_var_target(rho_1);
        let w_min_var_avg_23 = self.min_var_target(rho_avg_23);

        let alpha = (target_return - rho_avg_23) / (rho_1 - rho_avg_23);

        alpha * w_min_var_1 + (1.0 - alpha) * w_min_var_avg_23
    }

    pub fn tangent_portfolio(&self) -> DVector<f64> {
        let excess_rho = &self.rho - self.rho0 * &self.ones;
        let inv_cov_excess = &self.inv_cov * &excess_rho;

        let denominator = self.ones.dot(&inv_cov_excess);
        inv_cov_excess / denominator
    }

    pub fn one_fund_theorem_return(&self, target_return: f64) -> DVector<f64> {
        let tangent_portfolio = self.tangent_portfolio();
        let tangent_return = (self.rho.transpose() * &tangent_portfolio)[(0, 0)];

        let scaling_factor = target_return / tangent_return;
        scaling_factor * tangent_portfolio
    }

    pub fn one_fund_theorem_volatility(&self, target_volatility: f64) -> DVector<f64> {
        let tangent_portfolio = self.tangent_portfolio();
        let tangent_volatility =
            ((tangent_portfolio.transpose() * &self.cov * &tangent_portfolio)[(0, 0)]).sqrt();

        let scaling_factor = target_volatility / tangent_volatility;
        scaling_factor * tangent_portfolio
    }
}

pub fn a() {
    let cov = dmatrix![
        1.0, 0.2, 0.1;
        0.2, 1.1, 0.3;
        0.1, 0.3, 2.0
    ];
    let rho = dvector![4.27, 0.15, 2.85];
    let ones = DVector::from_element(rho.len(), 1.0);
    let inv_cov = cov
        .try_inverse()
        .expect("Covariance matrix is not invertible");

    let w_min_var = &inv_cov * &ones;
    let v = (ones.transpose() * &inv_cov * &ones)[(0, 0)];
    println!(
        "Global Minimum Variance Portfolio Weights: {}",
        w_min_var / v
    );

    let target_return = rho[0]; // Replace with your desired expected return
    let lambda = (target_return - (rho.transpose() * &inv_cov * &ones)[(0, 0)])
        / ((rho.transpose() * &inv_cov * &rho)[(0, 0)]
            - (rho.transpose() * &inv_cov * &ones)[(0, 0)]);

    let w_min_var_target = &inv_cov * (&ones + lambda * (&rho - &ones));
    println!(
        "Minimum Variance Portfolio Weights for Target Return {}: {}",
        target_return, w_min_var_target
    );
}
