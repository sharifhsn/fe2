use plotters::prelude::*;
pub struct OneFactorModel {
    pub a: f64,
    pub b: f64,
    pub sig: f64,
    pub r0: f64,
    pub model_type: OneFactorModelType,
}

pub enum OneFactorModelType {
    Vasicek,
    CoxIngersollRoss,
}

impl OneFactorModel {
    pub fn new(a: f64, b: f64, sig: f64, r0: f64, model_type: OneFactorModelType) -> Self {
        Self {
            a,
            b,
            sig,
            r0,
            model_type,
        }
    }

    pub fn P(&self, T: f64) -> f64 {
        match self.model_type {
            OneFactorModelType::Vasicek => {
                let B = (1.0 - (-self.a * T).exp()) / self.a;
                let A = ((self.b - self.sig.powi(2) / (2.0 * self.a.powi(2))) * (B - T)
                    - self.sig.powi(2) * B.powi(2) / (4.0 * self.a))
                    .exp();
                A * (-B * self.r0).exp()
            }
            OneFactorModelType::CoxIngersollRoss => {
                let h = (self.a.powi(2) + 2.0 * self.sig.powi(2)).sqrt();
                let A = ((2.0 * h * ((self.a + h) * T / 2.0).exp())
                    / (2.0 * h + (self.a + h) * ((T * h).exp() - 1.0)))
                    .powf(2.0 * self.a * self.b / self.sig.powi(2));
                let B = (2.0 * ((h * T).exp() - 1.0))
                    / (2.0 * h + (self.a + h) * ((T * h).exp() - 1.0));
                A * (-B * self.r0).exp()
            }
        }
    }

    pub fn y(&self, T: f64) -> f64 {
        (1.0 / self.P(T)).ln() / T
    }
}

pub fn a() {
    let a = 0.04;
    let b = 0.035;
    let sig = 0.04;
    let r0 = 0.045;
    let ts = (1..=5).map(|x| x as f64 * 2.0).collect::<Vec<_>>();
    let vasicek = OneFactorModel::new(a, b, sig, r0, OneFactorModelType::Vasicek);
    let cox_ingersoll_ross =
        OneFactorModel::new(a, b, sig, r0, OneFactorModelType::CoxIngersollRoss);
    println!("{:<10} {:<15} {:<15}", "T", "Vasicek", "CIR");
    for t in ts {
        let vasicek_price = vasicek.P(t);
        let cir_price = cox_ingersoll_ross.P(t);
        println!("{:<10.2} {:<15.6} {:<15.6}", t, vasicek_price, cir_price);
    }

    let root = BitMapBackend::new("vasicek_cir_price.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Vasicek and CIR Bond Prices", ("sans-serif", 50))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..10.0, 0.0..1.0)
        .unwrap();
    chart
        .configure_mesh()
        .x_desc("T")
        .y_desc("Price")
        .draw()
        .unwrap();

    let ts = (1..=1000).map(|x| x as f64 * 0.01).collect::<Vec<_>>();
    let vasicek_prices: Vec<f64> = ts.iter().map(|&t| vasicek.P(t)).collect();
    let cir_prices: Vec<f64> = ts.iter().map(|&t| cox_ingersoll_ross.P(t)).collect();

    chart
        .draw_series(LineSeries::new(
            ts.iter().zip(vasicek_prices.iter()).map(|(&t, &p)| (t, p)),
            &RED,
        ))
        .unwrap()
        .label("Vasicek")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(
            ts.iter().zip(cir_prices.iter()).map(|(&t, &p)| (t, p)),
            &BLUE,
        ))
        .unwrap()
        .label("CIR")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE)
        .draw()
        .unwrap();
}

pub fn b() {
    let a = 0.04;
    let b = 0.035;
    let sig = 0.04;
    let r0 = 0.045;
    let ts = (1..=5).map(|x| x as f64 * 2.0).collect::<Vec<_>>();
    let vasicek = OneFactorModel::new(a, b, sig, r0, OneFactorModelType::Vasicek);
    let cox_ingersoll_ross =
        OneFactorModel::new(a, b, sig, r0, OneFactorModelType::CoxIngersollRoss);
    println!("{:<10} {:<15} {:<15}", "T", "Vasicek", "CIR");
    for t in ts {
        let vasicek_yield = vasicek.y(t);
        let cir_yield = cox_ingersoll_ross.y(t);
        println!("{:<10.2} {:<15.6} {:<15.6}", t, vasicek_yield, cir_yield);
    }
    let root = BitMapBackend::new("vasicek_cir_yield.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Vasicek and CIR Yields", ("sans-serif", 50))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..10.0, 0.0..0.1)
        .unwrap();
    chart
        .configure_mesh()
        .x_desc("T")
        .y_desc("Yield")
        .draw()
        .unwrap();

    let ts = (1..=1000).map(|x| x as f64 * 0.01).collect::<Vec<_>>();
    let vasicek_yields: Vec<f64> = ts.iter().map(|&t| vasicek.y(t)).collect();
    let cir_yields: Vec<f64> = ts.iter().map(|&t| cox_ingersoll_ross.y(t)).collect();

    chart
        .draw_series(LineSeries::new(
            ts.iter().zip(vasicek_yields.iter()).map(|(&t, &y)| (t, y)),
            &RED,
        ))
        .unwrap()
        .label("Vasicek")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(
            ts.iter().zip(cir_yields.iter()).map(|(&t, &y)| (t, y)),
            &BLUE,
        ))
        .unwrap()
        .label("CIR")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE)
        .draw()
        .unwrap();
}
