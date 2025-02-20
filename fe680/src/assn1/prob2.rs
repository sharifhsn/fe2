use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
pub fn data_load() -> PolarsResult<DataFrame> {
    df!(
        "time_to_maturity" => [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0],
        "yield_rate" => [0.0150, 0.0160, 0.0180, 0.0210, 0.0240, 0.0330, 0.03740, 0.0405, 0.0435]
    )
}

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

    fn interpolated_rate(&self, t: f64) -> f64 {
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

use plotters::prelude::*;

fn plot_yield_curve(cs: &CubicSpline, x_data: &[f64], y_data: &[f64]) {
    let root = BitMapBackend::new("yield_curve.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_x = *x_data.first().unwrap();
    let max_x = *x_data.last().unwrap();

    let min_y = *y_data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_y = *y_data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Cubic Spline Yield Curve", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Compute interpolated values
    let x_fine: Vec<f64> = (0..100)
        .map(|i| min_x + i as f64 * (max_x - min_x) / 99.0)
        .collect();
    let y_fine: Vec<f64> = x_fine.iter().map(|&t| cs.interpolated_rate(t)).collect();

    // Draw the interpolated curve
    chart
        .draw_series(LineSeries::new(
            x_fine.iter().zip(y_fine.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))
        .unwrap()
        .label("Cubic Spline Interpolation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw original data points
    chart
        .draw_series(PointSeries::of_element(
            x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
            5,
            &RED,
            &|coord, size, style| {
                EmptyElement::at(coord) + Circle::new((0, 0), size, style.filled())
            },
        ))
        .unwrap()
        .label("Original Data Points")
        .legend(|(x, y)| Circle::new((x, y), 5, &RED));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()
        .unwrap();

    println!("Plot saved as 'yield_curve.png'");
}

pub fn a() -> PolarsResult<DataFrame> {
    let df = data_load()?;

    let x = df
        .column("time_to_maturity")?
        .f64()?
        .to_vec_null_aware()
        .left()
        .unwrap();
    let y = df
        .column("yield_rate")?
        .f64()?
        .to_vec_null_aware()
        .left()
        .unwrap();
    let cs = CubicSpline::new(x.clone(), y.clone());

    let r = cs.interpolated_rate(4.0);

    plot_yield_curve(&cs, &x, &y);
    Ok(df)
}
