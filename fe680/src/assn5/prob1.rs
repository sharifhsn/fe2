use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use statrs::distribution::{Normal as StatNormal, Continuous, ContinuousCDF};
use statrs::distribution::{Binomial, Discrete};
pub struct GaussianLatentVariableMultiName {
    pub Nc: usize,  // number of names
    pub R: f64,     // recovery rate
    pub beta: f64,  // market exposure
    pub p: f64,     // unconditional probability of default
    pub stat_normal: StatNormal, // normal distribution for Z
    pub normal: Normal<f64>, // normal distribution for Z
    pub z: f64,     // market factor Z (randomly generated)
    pub c_t: f64,   // calibrated time-dependent threshold C(T)
}

impl GaussianLatentVariableMultiName {
    // Constructor to initialize the model
    pub fn new(Nc: usize, R: f64, beta: f64, p: f64) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap(); // Mean 0, StdDev 1
        let stat_normal = StatNormal::new(0.0, 1.0).unwrap(); // Normal distribution for Z
        // Generate a random Z from a normal distribution
        let mut rng = rand::rng();
        let z = rng.sample(&normal); // Sample Z from N(0, 1)

        let c_t = stat_normal.inverse_cdf(p); // calibrate C(T) on initialization
        GaussianLatentVariableMultiName {
            Nc,
            R,
            beta,
            p,
            stat_normal, // Normal distribution for Z
            normal, // Normal distribution for Z
            z,
            c_t,
        }
    }

    // Compute P(T|Z) for each credit based on Z, beta, and C(T)
    fn p_t_given_z(&self, z: f64) -> f64 {
        self.stat_normal.cdf((self.c_t - self.beta * z) / (1.0 - self.beta.powi(2)).sqrt())
    }

    // Conditional loss distribution for each credit
    fn conditional_loss(&self, ptz: f64, n: usize) -> f64 {
        let binomial = Binomial::new(ptz, self.Nc as u64).unwrap(); // Binomial distribution for defaults
        binomial.pmf(n as u64)
    }

    // Unconditional loss distribution (integration over Z)
    pub fn unconditional_loss(&self, n: usize) -> f64 {
        // Fixed bounds for integration (from -5 to 5 for normal distribution approximation)
        let a = -5.0;
        let b = 5.0;
        
        // Number of steps for Simpson's quadrature
        let num_steps = 1000;

        // Simpson's quadrature
        let h = (b - a) / (num_steps as f64);

        // Function to represent the integrand
        let f = |z: f64| {
            let ptz = self.p_t_given_z(z); // Calculate P(T|Z)
            self.conditional_loss(ptz, n) * self.stat_normal.pdf(z) // Weight by the normal PDF
        };

        let sum = (1..num_steps)
            .into_iter()
            .map(|i| {
                let z_i = a + i as f64 * h; // Z value at this step
                let weight = if i % 2 == 0 { 2.0 } else { 4.0 }; // Simpson's rule weight
                weight * f(z_i) // Integrate the function
            })
            .sum::<f64>();

        // Final calculation using Simpson's rule
        (h / 3.0) * (f(a) + f(b) + sum)
    }

}

// Function to plot the unconditional loss distribution
pub fn plot_unconditional_loss(Nc: usize, R: f64, p: f64) {
    // Create the plot
    let root = BitMapBackend::new("unconditional_loss_distribution.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Unconditional Loss Distribution", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)

        .build_cartesian_2d(0.0..0.1, 0.0..0.2)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Portfolio Loss (%)")
        .y_desc("Probability")
        .draw()
        .unwrap();


    let portfolio4 = GaussianLatentVariableMultiName::new(
        Nc,   // Nc: number of credits
        R, // R: recovery rate
        0.4, // beta: market exposure
        p, // p: unconditional default probability
    );
    

    // Plot the unconditional loss for a range of n values
    let loss_values: Vec<(f64, f64)> = (0..=Nc)
        .map(|n| {
            let loss = portfolio4.unconditional_loss(n); // Calculate unconditional loss for each n
            (n as f64 / Nc as f64, loss)
        })
        .collect();

    // Draw the line plot
    chart
        .draw_series(LineSeries::new(loss_values, &BLUE))
        .unwrap()
        .label("β = 0.4")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], &BLUE));

    let portfolio2 = GaussianLatentVariableMultiName::new(
        Nc,   // Nc: number of credits
        R, // R: recovery rate
        0.2, // beta: market exposure
        p, // p: unconditional default probability
    );

    // Plot the unconditional loss for a range of n values
    let loss_values_beta2: Vec<(f64, f64)> = (0..=Nc)
        .map(|n| {
            let loss = portfolio2.unconditional_loss(n); // Calculate unconditional loss for each n
            (n as f64 / Nc as f64, loss)
        })
        .collect();

    // Draw the line plot
    chart
        .draw_series(LineSeries::new(loss_values_beta2, &RED))
        .unwrap()
        .label("β = 0.2")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], &RED));

    let portfolio0 = GaussianLatentVariableMultiName::new(
        Nc,   // Nc: number of credits
        R, // R: recovery rate
        0.0, // beta: market exposure
        p, // p: unconditional default probability
    );

    // Plot the unconditional loss for a range of n values
    let loss_values_beta0: Vec<(f64, f64)> = (0..=Nc)
        .map(|n| {
            let loss = portfolio0.unconditional_loss(n); // Calculate unconditional loss for each n
            (n as f64 / Nc as f64, loss)
        })
        .collect();

    // Draw the line plot
    chart
        .draw_series(LineSeries::new(loss_values_beta0, &GREEN))
        .unwrap()
        .label("β = 0.0")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], &GREEN));

    // Add the legend to the chart
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

pub fn a() {
    // Plot the unconditional loss distribution
    plot_unconditional_loss(100, 0.4, 0.04);
}
