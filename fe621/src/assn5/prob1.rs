use nalgebra::{dmatrix, dvector, DMatrix, DVector};
use ndarray::prelude::*;
use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
const EPS: f64 = 1e-6;
pub struct MyCholesky {
    pub A: DMatrix<f64>,
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        assert!(
            ($a - $b).iter().all(|&x| x.abs() < EPS),
            "Matrices are not approximately equal:\n{}\n{}",
            $a,
            $b
        );
    };
}

impl MyCholesky {
    pub fn new(A: DMatrix<f64>) -> Self {
        MyCholesky { A }
    }

    pub fn cholesky(&self) -> DMatrix<f64> {
        let n = self.A.nrows();
        let mut L = DMatrix::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += L[(i, k)] * L[(j, k)];
                }
                L[(i, j)] = if i == j {
                    (self.A[(i, j)] - sum).sqrt()
                } else {
                    (self.A[(i, j)] - sum) / L[(j, j)]
                };
            }
        }
        // basic approximate equality check
        assert_approx_eq!(L.clone() * L.transpose(), self.A.clone());
        // test against nalgebra's cholesky
        assert_eq!(L.clone(), self.A.clone().cholesky().unwrap().l());
        L
    }
}

pub fn a() {
    let A = dmatrix![1.0, 0.5, 0.2; 0.5, 1.0, -0.4; 0.2, -0.4, 1.0];
    let b = A.clone().cholesky();
    let chol = MyCholesky::new(A.clone());
    let L = chol.cholesky();
    println!("A = \n{}", A);
    println!("L = \n{}", L);
}

#[derive(Debug, Clone)]
pub struct Basket {
    pub S: DVector<f64>,
    pub mu: DVector<f64>,
    pub sig: DVector<f64>,
    pub A: DMatrix<f64>,
    pub T: f64,
    pub m: usize,
    pub dt: f64,
}

impl Basket {
    pub fn new(
        S: DVector<f64>,
        mu: DVector<f64>,
        sig: DVector<f64>,
        A: DMatrix<f64>,
        T: f64,
        m: usize,
        dt: f64,
    ) -> Self {
        Basket {
            S,
            mu,
            sig,
            A,
            T,
            m,
            dt,
        }
    }

    pub fn simulate(&self) -> Vec<DMatrix<f64>> {
        let n = self.S.len(); // number of assets
        let T = self.T as usize; // number of time steps
        let m = self.m; // number of paths

        // Cholesky decomposition of correlation matrix
        let L = self
            .A
            .clone()
            .cholesky() // I use nalgebra's cholesky for convenience
            .expect("Cholesky decomposition failed")
            .l();

        // Prepare RNGs per path
        let mut rngs: Vec<_> = (0..m).map(|_| rand::rng()).collect();

        // Output: one matrix per asset
        let mut results = vec![DMatrix::zeros(T, m); n];

        for path in 0..m {
            let rng = &mut rngs[path];
            // Initialize log prices for each asset
            let mut lnS: Vec<f64> = self.S.iter().map(|s| s.ln()).collect();

            // Set time 0
            for asset in 0..n {
                results[asset][(0, path)] = lnS[asset];
            }

            for t in 1..T {
                // Step 1: sample independent standard normals
                let mut eps = DVector::from_iterator(n, (0..n).map(|_| rng.sample(StandardNormal)));

                // Step 2: apply Cholesky to correlate
                let Z = &L * eps;

                // Step 3: evolve each asset
                for asset in 0..n {
                    let drift = (self.mu[asset] - 0.5 * self.sig[asset].powi(2)) * self.dt;
                    let diffusion = self.sig[asset] * Z[asset] * self.dt.sqrt();
                    lnS[asset] += drift + diffusion;

                    results[asset][(t, path)] = lnS[asset];
                }
            }
        }

        // Step 4: exponentiate in-place
        for matrix in &mut results {
            for x in matrix.iter_mut() {
                *x = x.exp();
            }
        }

        results
    }
}

pub fn b() {
    let S = dvector![100.0, 101.0, 98.0];
    let mu = dvector![0.03, 0.06, 0.02];
    let sig = dvector![0.05, 0.2, 0.15];
    let A = dmatrix![1.0, 0.5, 0.2; 0.5, 1.0, -0.4; 0.2, -0.4, 1.0];
    let T = 100.0;
    let m = 1000;
    let dt = 1.0 / 365.0;
    let basket = Basket::new(S, mu, sig, A, T, m, dt);
    let results = basket.simulate();
    let root = BitMapBackend::new("simulation_paths.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Simulated Paths", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..basket.T as usize, 80.0..120.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    for (idx, matrix) in results.iter().enumerate() {
        let path: Vec<(usize, f64)> = matrix
            .column(0)
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        chart
            .draw_series(LineSeries::new(
                path,
                &Palette99::pick(idx % Palette99::COLORS.len() + 3),
            ))
            .unwrap()
            .label(format!("Asset {}", idx + 1))
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    &Palette99::pick(idx % Palette99::COLORS.len() + 3),
                )
            });
    }

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

#[derive(Debug, Clone, Copy)]
pub enum OptionType {
    Call,
    Put,
}

pub struct BasketEuropeanOption {
    pub basket: Basket,
    pub K: f64,
    pub a: DVector<f64>, // weights
    pub option_type: OptionType,
}

impl BasketEuropeanOption {
    pub fn new(basket: Basket, K: f64, a: DVector<f64>, option_type: OptionType) -> Self {
        BasketEuropeanOption {
            basket,
            K,
            a,
            option_type,
        }
    }

    pub fn price(&self) -> f64 {
        let sim = self.basket.simulate();
        sim.iter()
            .enumerate()
            .map(|(i, matrix)| {
                let S = matrix.column_mean()[self.basket.T as usize - 1];
                let payoff = match self.option_type {
                    OptionType::Call => (S - self.K).max(0.0),
                    OptionType::Put => (self.K - S).max(0.0),
                } * self.a[i];
                payoff
            })
            .sum()
    }
}

pub fn c() {
    let S = dvector![100.0, 101.0, 98.0];
    let mu = dvector![0.03, 0.06, 0.02];
    let sig = dvector![0.05, 0.2, 0.15];
    let A = dmatrix![1.0, 0.5, 0.2; 0.5, 1.0, -0.4; 0.2, -0.4, 1.0];
    let T = 100.0;
    let m = 1_000_000;
    let dt = 1.0 / 365.0;
    let basket = Basket::new(S, mu, sig, A, T, m, dt);
    let K = 100.0;
    let a = dvector![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let call_option = BasketEuropeanOption::new(basket.clone(), K, a.clone(), OptionType::Call);
    println!("Call Option Price: {}", call_option.price());
    let put_option = BasketEuropeanOption::new(basket.clone(), K, a.clone(), OptionType::Put);
    println!("Put Option Price: {}", put_option.price());
}

pub struct BasketExoticOption {
    pub basket: Basket,
    pub K: f64,
    pub B: f64,
    pub a: DVector<f64>, // weights
}

impl BasketExoticOption {
    pub fn new(basket: Basket, K: f64, B: f64, a: DVector<f64>) -> Self {
        BasketExoticOption { basket, K, B, a }
    }

    pub fn price(&self) -> f64 {
        let sim = self.basket.simulate();
        let mut results = vec![];
        for i in 0..self.basket.m {
            // (i)
            let ST2 = sim[1].column(i)[self.basket.T as usize - 1];
            let max2 = sim[1].column(i).max();

            // (ii)
            let max3 = sim[2].column(i).max();

            // (iii)
            let AT2 = sim[1].column(i).mean();
            let AT3 = sim[2].column(i).mean();

            let payoff = if max2 > self.B {
                (ST2 - self.K).max(0.0)
            } else if max2 > max3 {
                (ST2.powi(2) - self.K).max(0.0)
            } else if AT2 > AT3 {
                (AT2 - self.K).max(0.0)
            } else {
                let mut sum = 0.0;
                for i in 0..sim.len() {
                    let ST = sim[i].column(i)[self.basket.T as usize - 1];
                    sum += (ST - self.K).max(0.0) * self.a[i];
                }
                sum
            };
            results.push(payoff);
        }
        results.into_iter().sum::<f64>() / self.basket.m as f64
    }
}

pub fn d() {
    let S = dvector![100.0, 101.0, 98.0];
    let mu = dvector![0.03, 0.06, 0.02];
    let sig = dvector![0.05, 0.2, 0.15];
    let A = dmatrix![1.0, 0.5, 0.2; 0.5, 1.0, -0.4; 0.2, -0.4, 1.0];
    let T = 100.0;
    let m = 1_000_000;
    let dt = 1.0 / 365.0;
    let basket = Basket::new(S, mu, sig, A, T, m, dt);
    let K = 100.0;
    let B = 104.0;
    let a = dvector![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let exotic_option = BasketExoticOption::new(basket.clone(), K, B, a.clone());
    println!("Exotic Option Price: {}", exotic_option.price());
}
