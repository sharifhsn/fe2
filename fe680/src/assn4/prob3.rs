use nalgebra::{DMatrix, DVector};
use plotters::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

const R: f64 = 0.4;
const BP: f64 = 0.0001;

fn d_from_p(y: f64, t: f64) -> f64 {
    (-y / 100.0 * t).exp()
}

pub struct CreditDefaultSwap {
    pub cs: CubicSpline,
    pub T: Vec<f64>,
    pub s: f64,
}

impl CreditDefaultSwap {
    pub fn new(cs: CubicSpline, T: Vec<f64>, s: f64) -> Self {
        CreditDefaultSwap { cs, T, s }
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
        let pv_prem: f64 = (1..self.T.len()).map(|i| self.cs.P(self.T[i]) * Q[i]).sum();

        // PV protection
        let pv_protection: f64 = (1..self.T.len())
            .map(|i| self.cs.P((self.T[i - 1] + self.T[i]) / 2.0) * D[i] * (1.0 - R))
            .sum();

        // PV premium accrual
        let pv_accrual: f64 = (1..self.T.len())
            .map(|i| self.cs.P((self.T[i - 1] + self.T[i]) / 2.0) * D[i])
            .sum();

        let s = pv_protection / (pv_prem + pv_accrual);
        self.s - s
    }

    fn bisect(&self, mut a: f64, mut b: f64, ε: f64, max_iter: usize) -> Option<f64> {
        if (self.hazard_to_spread(a) * self.hazard_to_spread(b)).is_sign_positive() {
            eprintln!("f(a) and f(b) must have opposite signs");
            return None;
        }

        let mut mid;
        for _ in 0..max_iter {
            mid = (a + b) / 2.0;

            if self.hazard_to_spread(mid).abs() < ε {
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

pub fn a() {
    let M = 1.0 / 12.0;
    let tenors = vec![
        1.0 * M,
        2.0 * M,
        3.0 * M,
        6.0 * M,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ];
    let yields = vec![
        1.495, 1.5372, 1.6696, 1.9304, 2.2765, 2.0635, 2.2165, 2.339, 2.4365, 2.524, 2.597, 2.661,
        2.7205, 2.767,
    ];
    let cs = CubicSpline::new(tenors.clone(), yields.clone());

    let cds_tenors = vec![0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0];
    let cds_spreads = vec![40.2, 48.9, 57.1, 68.8, 82.8, 93.9, 119.8, 137.7];

    let hazards: Vec<f64> = cds_tenors
        .iter()
        .zip(cds_spreads.iter())
        .map(|(&tenor, &spread)| {
            let T: Vec<f64> = (1..=(tenor * 4.0) as usize)
                .map(|x| x as f64 * 0.25)
                .collect();
            let cds = CreditDefaultSwap::new(cs.clone(), T, spread * BP);
            cds.bisect(1e-5, 1.0, 1e-10, 1000).unwrap()
        })
        .collect();
    let hazard_cs = CubicSpline::new(cds_tenors.clone(), hazards.clone());
    println!("hazards: {:.06?}", hazards);

    // integrate
    let Q = |t| (-inthazard(hazard_cs.clone(), 0.0, t, 100_000)).exp();
    let D = |t| 1.0 - Q(t);

    let ts: Vec<f64> = (0..=120).map(|i| i as f64 / 12.0).collect(); // 0.0 to 10.0
    let mut dtimes = vec![];

    for &t in &ts {
        dtimes.push((t, D(t)));
    }

    let root = BitMapBackend::new("cumulative_default.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Cumulative Default Probability", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..10f64, 0f64..0.5)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Years")
        .y_desc("D(t)")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(dtimes.iter().copied(), &BLUE))
        .unwrap();

    println!("Plot saved to cumulative_default.png");
}

/// Integral of the hazard function
fn inthazard(cs: CubicSpline, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / (n as f64);

    let sum: f64 = (1..n)
        .into_par_iter()
        .map(|i| {
            let x_i = -a + i as f64 * h;
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            weight * cs.r(x_i)
        })
        .sum();

    (h / 3.0) * (cs.r(a) + cs.r(b) + sum)
}

#[derive(Clone)]
struct CubicSpline {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSpline {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n = x.len();
        assert!(n > 2, "At least two data points are required");

        let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let m = Self::second_derivatives(&x, &y, &h);
        let (a, b, c, d) = Self::compute_coefficients(&x, &y, &h, &m);

        Self { x, a, b, c, d }
    }

    fn second_derivatives(x: &[f64], y: &[f64], h: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut a = DMatrix::zeros(n, n);
        let mut b = DVector::zeros(n);

        // Natural spline boundary conditions (M_0 = 0, M_n-1 = 0)
        a[(0, 0)] = 1.0;
        a[(n - 1, n - 1)] = 1.0;

        for i in 1..n - 1 {
            a[(i, i - 1)] = h[i - 1];
            a[(i, i)] = 2.0 * (h[i - 1] + h[i]);
            a[(i, i + 1)] = h[i];
            b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }

        let m = a.lu().solve(&b).expect("Failed to solve system");
        m.as_slice().to_vec()
    }

    fn compute_coefficients(
        x: &[f64],
        y: &[f64],
        h: &[f64],
        m: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = x.len() - 1;
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];

        for i in 0..n {
            a[i] = y[i];
            b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 6.0) * (m[i + 1] + 2.0 * m[i]);
            c[i] = m[i] / 2.0;
            d[i] = (m[i + 1] - m[i]) / (6.0 * h[i]);
        }

        (a, b, c, d)
    }

    fn r(&self, t: f64) -> f64 {
        if t > self.x[self.x.len() - 1] {
            panic!("Extrapolation is not supported!");
        }

        let i = self
            .x
            .partition_point(|&xi| xi <= t)
            .saturating_sub(1)
            .min(self.x.len() - 2);

        let dx = t - self.x[i];
        self.a[i] + self.b[i] * dx + self.c[i] * dx.powi(2) + self.d[i] * dx.powi(3)
    }

    fn P(&self, t: f64) -> f64 {
        (-self.r(t) * t).exp()
    }
}
