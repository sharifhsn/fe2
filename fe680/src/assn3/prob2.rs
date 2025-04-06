use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};

use argmin::{
    core::{CostFunction, Executor},
    solver::neldermead::NelderMead,
};

const BP: f64 = 0.0001;

struct ProbSystem {
    E: f64,
    Ex: f64,
    stdev: f64,
}

impl ProbSystem {
    fn new(E: f64, Ex: f64, stdev: f64) -> Self {
        Self { E, Ex, stdev }
    }
}

impl CostFunction for ProbSystem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        let p = param[0];
        let r = param[1];

        let E = self.E;
        let Ex = self.Ex;
        let stdev = self.stdev;

        let e1 = p * r + (1.0 - p) * Ex - E;
        let e2 = p * (r - E).powi(2) + (1.0 - p) * (Ex - E).powi(2) - stdev.powi(2);

        Ok(e1.powi(2) + e2.powi(2))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OptionStyle {
    European,
    American,
}

#[derive(Clone, Copy)]
enum OptionType {
    Call,
    Put,
}

#[derive(Clone, Copy)]
struct Zcb {
    L: f64,
    T: f64,
}

#[derive(Clone, Copy)]
struct ZcbOption {
    zcb: Zcb,
    K: f64,
    option_type: OptionType,
    option_style: OptionStyle,
    T: f64,
}

struct VasicekBinomialTree {
    r0: f64,    // Initial short rate
    theta: f64, // Long-term mean
    k: f64,     // Mean reversion speed
    sigma: f64, // Volatility
    dt: f64,    // Time step
    T: f64,     // Time to maturity
    n: usize,   // Number of steps
    tree: DMatrix<f64>,
    ptree: DMatrix<f64>,
}

impl VasicekBinomialTree {
    fn new(r0: f64, theta: f64, k: f64, sigma: f64, dt: f64, T: f64) -> Self {
        let n = (T / dt) as usize;
        let stdev = sigma * (dt).sqrt();

        // Solver for probability systems of equations
        let init_param = vec![0.5, r0];
        let simplex = vec![
            init_param.clone(),
            vec![init_param[0] + 0.1, init_param[1]],
            vec![init_param[0], init_param[1] + 0.1],
        ];
        let solver = NelderMead::new(simplex);

        let edr = |r: f64| r + k * (theta - r) * dt;
        let mut expectations = vec![r0];
        for i in 1..=n {
            expectations.push(edr(expectations[i - 1]));
        }

        // Initialize trees
        let mut ptree = DMatrix::zeros(2 * n, n + 1);
        ptree[(0, 0)] = 1.0;
        ptree[(0, 1)] = 0.5;
        ptree[(1, 1)] = 0.5;

        let mut tree = DMatrix::zeros(n + 1, n + 1);
        tree[(0, 0)] = r0;
        tree[(0, 1)] = expectations[1] + stdev;
        tree[(1, 1)] = expectations[1] - stdev;

        for i in 2..=n {
            if i % 2 == 0 {
                tree[(i / 2, i)] = expectations[i];
            } else {
                tree[(i / 2, i)] = expectations[i] + stdev;
                tree[(i / 2 + 1, i)] = expectations[i] - stdev;

                ptree[(i - 1, i)] = 0.5;
                ptree[(i, i)] = 0.5;
            }

            for j in (0..i / 2).rev() {
                let Ex = tree[(j + 1, i)];
                let E = edr(tree[(j, i - 1)]);
                let (p, r) = solve(ProbSystem::new(E, Ex, stdev), solver.clone());
                ptree[(j * 2, i)] = p;
                ptree[(j * 2 + 1, i)] = 1.0 - p;
                tree[(j, i)] = r;
            }

            for j in 1 + ((i + 1) / 2)..=i {
                let Ex = tree[(j - 1, i)];
                let E = edr(tree[(j - 1, i - 1)]);
                let (q, r) = solve(ProbSystem::new(E, Ex, stdev), solver.clone());
                ptree[(2 * (j - 1), i)] = 1.0 - q;
                ptree[(2 * (j - 1) + 1, i)] = q;
                tree[(j, i)] = r;
            }
        }
        println!("{}", tree);
        println!("{}", ptree);

        Self {
            r0,
            theta,
            k,
            sigma,
            dt,
            T,
            n,
            tree,
            ptree,
        }
    }

    fn terminal_rates(&self) -> Vec<f64> {
        self.tree
            .column(self.n)
            .iter()
            .cloned()
            .collect::<Vec<f64>>()
    }

    fn zcb(&self, zcb: Zcb) -> DMatrix<f64> {
        let mut bond_tree = DMatrix::zeros(self.n + 1, self.n + 1);
        let n = (zcb.T / self.dt) as usize;
        for i in 0..=n {
            bond_tree[(i, n)] = zcb.L;
        }

        for i in (0..n).rev() {
            for j in 0..=i {
                let r = self.tree[(j, i)];
                let pu = self.ptree[(2 * j, i + 1)];
                let pd = self.ptree[(2 * j + 1, i + 1)];
                bond_tree[(j, i)] = (-r * self.dt).exp()
                    * (pu * bond_tree[(j, i + 1)] + pd * bond_tree[(j + 1, i + 1)]);
            }
        }

        bond_tree
    }

    fn zcb_option(&self, zcb_option: ZcbOption) -> DMatrix<f64> {
        let mut option_tree = DMatrix::zeros(self.n + 1, self.n + 1);
        let n = (zcb_option.T / self.dt) as usize;
        let bond_tree = self.zcb(zcb_option.zcb);
        for i in 0..=n {
            let B = bond_tree[(i, n)];
            option_tree[(i, n)] = match zcb_option.option_type {
                OptionType::Call => (B - zcb_option.K).max(0.0),
                OptionType::Put => (zcb_option.K - B).max(0.0),
            };
        }

        for i in (0..n).rev() {
            for j in 0..=i {
                let r = self.tree[(j, i)];
                let pu = self.ptree[(2 * j, i + 1)];
                let pd = self.ptree[(2 * j + 1, i + 1)];

                option_tree[(j, i)] = (-r * self.dt).exp()
                    * (pu * option_tree[(j, i + 1)] + pd * option_tree[(j + 1, i + 1)]);

                if zcb_option.option_style == OptionStyle::American {
                    let B = bond_tree[(j, i)];
                    let payoff = match zcb_option.option_type {
                        OptionType::Call => (B - zcb_option.K).max(0.0),
                        OptionType::Put => (zcb_option.K - B).max(0.0),
                    };
                    option_tree[(j, i)] = payoff.max(option_tree[(j, i)]);
                }
            }
        }

        option_tree
    }
}

fn solve(problem: ProbSystem, solver: NelderMead<Vec<f64>, f64>) -> (f64, f64) {
    let res = Executor::new(problem, solver).run().unwrap();
    let state = res.state().best_param.clone().unwrap();
    (state[0], state[1])
}

pub fn a() {
    let k = 0.025;
    let dt = 1.0 / 12.0;
    let T = 6.0 / 12.0;
    let sigma = 126.0 * BP;
    let r0 = 0.05121;
    let theta = 0.15339;

    let L = 100.0;
    let zcb = Zcb { L, T };
    let zcb_eur_option = ZcbOption {
        zcb,
        K: 98.0,
        option_type: OptionType::Call,
        option_style: OptionStyle::European,
        T: 3.0 / 6.0,
    };
    let zcb_amr_option = ZcbOption {
        zcb,
        K: 98.0,
        option_type: OptionType::Call,
        option_style: OptionStyle::American,
        T: 3.0 / 6.0,
    };

    let tree = VasicekBinomialTree::new(r0, theta, k, sigma, dt, T);
    println!("{:?}", tree.terminal_rates());
    println!("{}", tree.zcb(zcb)[(0, 0)]);
    println!("{}", tree.zcb_option(zcb_eur_option)[(0, 0)]);
    println!("{}", tree.zcb_option(zcb_amr_option)[(0, 0)]);
}
