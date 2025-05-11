use nalgebra::DMatrix;
use statrs::distribution::{ContinuousCDF, Normal};
use rand_distr::StandardNormal;
use rand::prelude::*;
use rayon::prelude::*;
use plotters::prelude::*;

const BP: f64 = 0.0001;

pub struct GaussianTwoFactor {
    pub Nc: usize,
    pub beta1: Vec<f64>,
    pub beta2: Vec<f64>,
    pub hazards: Vec<f64>,
    pub m: usize,
}

pub struct SummaryStatistics {
    min: f64,
    q1: f64,
    median: f64,
    mean: f64,
    q3: f64,
    max: f64
}

impl GaussianTwoFactor {
    pub fn new(Nc: usize, beta1: Vec<f64>, beta2: Vec<f64>, hazards: Vec<f64>, m: usize) -> Self {
        GaussianTwoFactor {
            Nc,
            beta1,
            beta2,
            hazards,
            m,
        }
    }

    pub fn simulate(&self) -> DMatrix<f64> {
        let mut M = DMatrix::zeros(self.Nc, self.m);
        M.par_column_iter_mut().for_each(|mut col| {
            let mut rng = rand::rng();
            let Z1: f64 = rng.sample(StandardNormal);
            let Z2: f64 = rng.sample(StandardNormal);
            for i in 0..self.Nc {
                let beta1 = self.beta1[i];
                let beta2 = self.beta2[i];
                let lambda = self.hazards[i];
                let epsilon: f64 = rng.sample(StandardNormal);
                let A = beta1 * Z1 + beta2 * Z2 + (1.0 - beta1.powi(2) - beta2.powi(2)).sqrt() * epsilon;
                col[i] = -(1.0 - Normal::standard().cdf(A)).ln() / lambda;
            }
        });
        M
    }

    pub fn summary_statistics(&self, matrix: &DMatrix<f64>) -> Vec<SummaryStatistics> {
        matrix.row_iter()
            .map(|row| {
                let mut sorted_row: Vec<f64> = row.iter().cloned().collect();
                sorted_row.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let len = sorted_row.len();
                let mean = row.mean();
                let min = sorted_row[0];
                let max = sorted_row[len - 1];
                let median = if len % 2 == 0 {
                    (sorted_row[len / 2 - 1] + sorted_row[len / 2]) / 2.0
                } else {
                    sorted_row[len / 2]
                };
                let q1 = sorted_row[len / 4];
                let q3 = sorted_row[3 * len / 4];
                SummaryStatistics {
                    min,
                    q1,
                    median,
                    mean,
                    q3,
                    max,
                }
            })
            .collect()
    }
}

pub fn a() {
    let Nc = 10;
    let beta1 = vec![0.15; Nc];
    let beta2 = [vec![-0.6; Nc / 2], vec![0.4; Nc / 2]].concat();
    let hazards = [vec![150.0 * BP; Nc / 2], vec![300.0 * BP; Nc / 2]].concat();
    let m = 1_000_000;
    let g2f = GaussianTwoFactor::new(Nc, beta1, beta2, hazards, m);
    let tau = g2f.simulate();
    println!("Tau: {}", tau.column_mean());
}

pub fn b() {
    let Nc = 10;
    let beta1 = vec![0.15; Nc];
    let beta2 = [vec![-0.6; Nc / 2], vec![0.4; Nc / 2]].concat();
    let hazards = [vec![150.0 * BP; Nc / 2], vec![300.0 * BP; Nc / 2]].concat();
    let m = 1_000_000;
    let g2f = GaussianTwoFactor::new(Nc, beta1, beta2, hazards, m);
    let tau = g2f.simulate();
    let stats = g2f.summary_statistics(&tau);
    for (i, stat) in stats.iter().enumerate() {
        println!(
            "Credit {}: min = {:.4}, q1 = {:.4}, median = {:.4}, mean = {:.4}, q3 = {:.4}, max = {:.4}",
            i + 1, stat.min, stat.q1, stat.median, stat.mean, stat.q3, stat.max
        );
    }
}

pub fn c() {
    let Nc = 10;
    let beta1 = vec![0.15; Nc];
    let beta2 = [vec![-0.6; Nc / 2], vec![0.4; Nc / 2]].concat();
    let hazards = [vec![150.0 * BP; Nc / 2], vec![300.0 * BP; Nc / 2]].concat();
    let m = 1_000_000;
    let g2f = GaussianTwoFactor::new(Nc, beta1, beta2, hazards, m);
    let tau = g2f.simulate();
    for (i, credit) in tau.row_iter().enumerate() {
        let mut trials = credit.iter().cloned().collect::<Vec<f64>>();
        trials.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_trial = trials[0];
        let max_trial = trials[trials.len() - 1];
        let num_bins = 100;
        let bin_width = (max_trial - min_trial) / num_bins as f64;
        let mut pdf = vec![0.0; num_bins];

        for &value in &trials {
            let bin = ((value - min_trial) / bin_width).floor() as usize;
            if bin < num_bins {
                pdf[bin] += 1.0;
            }
        }

        for count in &mut pdf {
            *count /= trials.len() as f64 * bin_width;
        }
        use std::io::Write;
        let file_name = format!("credit_{}_pdf.txt", i + 1);
        let mut file = std::fs::File::create(&file_name).unwrap();
        writeln!(file, "Credit {} Times\tPDF Value", i + 1).unwrap();
        for (i, &value) in pdf.iter().enumerate() {
            let bin_start = min_trial + i as f64 * bin_width;
            let bin_end = bin_start + bin_width;
            writeln!(file, "[{:.4}, {:.4})\t{:.6}", bin_start, bin_end, value).unwrap();
        }


        let s = format!("credit_{}_pdf.png", i + 1);
        let root = BitMapBackend::new(&s, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Probability Density Function of Credit {}", i + 1), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_trial..300.0, 0.0..pdf.iter().cloned().fold(0.0, f64::max))
            .unwrap();

        chart.configure_mesh().draw().unwrap();
        chart
            .draw_series(
            pdf.iter()
                .enumerate()
                .map(|(i, &value)| {
                let x0 = min_trial + i as f64 * bin_width;
                let x1 = x0 + bin_width;
                Rectangle::new([(x0, 0.0), (x1, value)], BLUE.filled())
                }),
            )
            .unwrap();
    }
}