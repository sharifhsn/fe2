use std::sync::LazyLock;
use statrs::distribution::{Normal, ContinuousCDF};

/// Lazily‐initialized standard normal (mean=0, σ=1).
static STANDARD_NORMAL: LazyLock<Normal> =
    LazyLock::new(|| Normal::new(0.0, 1.0).expect("Failed to create standard normal"));

/// A minimal bivariate‐normal struct
/// 
/// Only supports CDF(X≤x, Y≤y) for correlation ρ.
pub struct BvNormal;

impl BvNormal {
    /// Create a new bivariate normal distribution.
    pub fn new() -> Self {
        BvNormal
    }
    
    /// Returns P(X ≤ x, Y ≤ y) under N₂(0,0,1,1,ρ).
    pub fn cdf(&self, x: f64, y: f64, rho: f64) -> f64 {
        // bring ρ into local scope
        let abs_rho = rho.abs();

        // constants
        const TAU: f64 = std::f64::consts::TAU;             // 2π
        let SQRT_TWO_PI: f64 = TAU.sqrt();                // √(2π)

        // Gauss–Legendre nodes & weights (columns: |r|<0.3, <0.75, ≥0.75)
        const X: [[f64; 3]; 10] = [
            [-0.9324695142031522, -0.9815606342467191, -0.9931285991850949],
            [-0.6612093864662647, -0.9041172563704750, -0.9639719272779138],
            [-0.2386191860831970, -0.7699026741943050, -0.9122344282513259],
            [ 0.2386191860831970, -0.5873179542866171, -0.8391169718222188],
            [ 0.6612093864662647, -0.3678314989981802, -0.7463319064601508],
            [ 0.9324695142031522, -0.1252334085114692, -0.6360536807265150],
            [ 0.0,                0.0,                -0.5108670019508271],
            [ 0.0,                0.0,                -0.3737060887154196],
            [ 0.0,                0.0,                -0.2277858511416451],
            [ 0.0,                0.0,                -0.0765265211334973],
        ];
        const W: [[f64; 3]; 10] = [
            [0.1713244923791705, 0.04717533638651177, 0.01761400713915212],
            [0.3607615730481384, 0.1069393259953183,  0.04060142980038694],
            [0.4679139345726904, 0.1600783285433464,  0.06267204833410906],
            [0.0,                0.2031674267230659,  0.08327674157670475],
            [0.0,                0.2334925365383547,  0.1019301198172404 ],
            [0.0,                0.2491470458134029,  0.1181945319615184 ],
            [0.0,                0.0,                 0.1316886384491766 ],
            [0.0,                0.0,                 0.1420961093183821 ],
            [0.0,                0.0,                 0.1491729864726037 ],
            [0.0,                0.0,                 0.1527533871307259 ],
        ];

        // pick which column & how many points
        let (col, pts) = match abs_rho {
            a if a < 0.3  => (0, 3),
            a if a < 0.75 => (1, 6),
            _             => (2, 10),
        };

        // “Main” vs. “tail” region switch
        if abs_rho < 0.925 {
            // Main region
            // hs = (x^2 + y^2)/2
            let hs = 0.5 * x.mul_add(x, y*y);
            let asr = rho.asin();
            let mut acc = 0.0;

            for i in 0..pts {
                let xi = X[i][col];
                let wi = W[i][col];
                // first node
                let sn1 = ((xi + 1.0) * asr * 0.5).sin();
                acc += wi * ((sn1.mul_add(x*y, -hs)) / (1.0 - sn1*sn1)).exp();
                // mirrored node
                let sn2 = ((-xi + 1.0) * asr * 0.5).sin();
                acc += wi * ((sn2.mul_add(x*y, -hs)) / (1.0 - sn2*sn2)).exp();
            }

            // complete with univariate tails
            acc * (asr / TAU) + 
            STANDARD_NORMAL.cdf(-x) * STANDARD_NORMAL.cdf(-y)
        } else {
            // Tail region
            let hh = x;
            let kk = if rho < 0.0 { -y } else { y };

            let mut term = 0.0;
            if abs_rho < 1.0 {
                let asq = (1.0 - rho).mul_add(1.0 + rho, 0.0);     // (1−r)*(1+r)
                let a   = asq.sqrt();
                let bs  = (hh - kk).powi(2);
                let c   = (4.0 - hh*kk) / 8.0;
                let d   = (12.0 - hh*kk) / 16.0;

                // first piece
                let mut t0 = a * (-(bs/asq + hh*kk)/2.0).exp()
                    * (1.0 - c*(bs - asq)*(1.0 - d*bs/5.0)/3.0
                       + c*d*asq*asq/5.0);

                // correction if not too extreme
                if hh*kk > -160.0 {
                    let b = bs.sqrt();
                    t0 -= (-(hh*kk)/2.0).exp()
                        * SQRT_TWO_PI
                        * STANDARD_NORMAL.cdf(-b / a)
                        * b
                        * (1.0 - c*bs*(1.0 - d*bs/5.0)/3.0);
                }

                // the double‐integral correction
                let ap2 = 0.5 * a;
                for i in 0..pts {
                    let xi = X[i][col];
                    let wi = W[i][col];

                    // part 1
                    let xs1 = (ap2.mul_add(xi+1.0, 0.0)).powi(2);
                    let rs1 = (1.0 - xs1).sqrt();
                    t0 += ap2 * wi * 
                        ((-(bs/(2.0*xs1)) - hh*kk/(1.0+rs1)).exp() / rs1
                         - (-(bs/xs1 + hh*kk)/2.0).exp() * (1.0 + c*xs1*(1.0 + d*xs1)));

                    // part 2
                    let xs2 = asq * (-xi + 1.0).powi(2) * 0.25;
                    let rs2 = (1.0 - xs2).sqrt();
                    t0 += ap2 * wi *
                        ( (-(bs/xs2 + hh*kk)/2.0).exp()
                          * ((-hh*kk*(1.0 - rs2)/(2.0*(1.0 + rs2))).exp()/rs2
                             - (1.0 + c*xs2*(1.0 + d*xs2))) );
                }

                term = -t0 / TAU;
            }

            // final adjustment for sign of ρ
            if rho > 0.0 {
                term + STANDARD_NORMAL.cdf(-x.max(y))
            } else {
                -term + (STANDARD_NORMAL.cdf(-x) - STANDARD_NORMAL.cdf(-y)).max(0.0)
            }
        }
    }
}
