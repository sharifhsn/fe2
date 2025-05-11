use nalgebra::{dmatrix, dvector, DMatrix, DVector};

const BP: f64 = 0.0001;

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

    fn y(&self, t: f64) -> f64 {
        if t < self.x[0] || t > self.x[self.x.len() - 1] {
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
}

fn P(y: f64, t: f64) -> f64 {
    -(y * t).exp()
}

pub trait Bond {
    /// bond price
    fn B(&self, t: f64) -> f64;
    /// modified duration
    fn D(&self, t: f64) -> f64;
    /// convexity
    fn C(&self, t: f64) -> f64;
}

impl Bond for CubicSpline {
    fn B(&self, t: f64) -> f64 {
        -P(self.y(t), t)
    }

    fn D(&self, t: f64) -> f64 {
        -(P(self.y(t) + BP, t) - P(self.y(t) - BP, t)) / (2.0 * self.B(t) * BP)
    }

    fn C(&self, t: f64) -> f64 {
        (P(self.y(t) + BP, t) + P(self.y(t) - BP, t) - 2.0 * self.B(t)) / (self.B(t) * BP.powi(2))
    }
}

pub fn a() {
    let times = vec![0.5, 10.0, 30.0];
    let yields = vec![0.0423, 0.0428, 0.0477];

    let cs = CubicSpline::new(times, yields);
    let y = cs.y(7.0);
    println!("yield: {}", y);
}

pub fn b() {
    let times = vec![0.5, 10.0, 30.0];
    let yields = vec![0.0423, 0.0428, 0.0477];

    let cs = CubicSpline::new(times, yields);
    let times = [2.0, 0.5, 10.0, 30.0];
    let bond_prices = times
        .iter()
        .map(|&t| (t, cs.B(t)))
        .collect::<Vec<(f64, f64)>>();
    for (time, price) in bond_prices {
        println!("time: {}, bond price: {}", time, price);
    }

    // create a matrix for optimization
    let A = dmatrix![
        cs.B(0.5), cs.B(10.0), cs.B(30.0);
        cs.D(0.5), cs.D(10.0), cs.D(30.0);
        cs.C(0.5), cs.C(10.0), cs.C(30.0)
    ];
    let b = dvector![cs.B(7.0), -cs.D(7.0), -cs.C(7.0)];
    let x = A.lu().solve(&b).expect("Failed to solve system");
    println!("x: {:.4}", x);
}
