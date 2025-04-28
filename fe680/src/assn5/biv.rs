#![allow(clippy::excessive_precision)]

use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::TAU as TWOPI;
use std::sync::LazyLock;

/// Lazily‐initialized standard normal (mean=0, σ=1).
static STANDARD_NORMAL: LazyLock<Normal> =
    LazyLock::new(|| Normal::new(0.0, 1.0).expect("Failed to create standard normal"));

// Gauss–Legendre nodes & weights (columns: |r|<0.3, <0.75, ≥0.75)
const X: [[f64; 3]; 10] = [
    [
        -0.9324695142031522,
        -0.9815606342467191,
        -0.9931285991850949,
    ],
    [
        -0.6612093864662647,
        -0.9041172563704750,
        -0.9639719272779138,
    ],
    [
        -0.2386191860831970,
        -0.7699026741943050,
        -0.9122344282513259,
    ],
    [0.0, -0.5873179542866171, -0.8391169718222188],
    [0.0, -0.3678314989981802, -0.7463319064601508],
    [0.0, -0.1252334085114692, -0.6360536807265150],
    [0.0, 0.0, -0.5108670019508271],
    [0.0, 0.0, -0.3737060887154196],
    [0.0, 0.0, -0.2277858511416451],
    [0.0, 0.0, -0.0765265211334973],
];
const W: [[f64; 3]; 10] = [
    [0.1713244923791705, 0.04717533638651177, 0.01761400713915212],
    [0.3607615730481384, 0.1069393259953183, 0.04060142980038694],
    [0.4679139345726904, 0.1600783285433464, 0.06267204833410906],
    [0.0, 0.2031674267230659, 0.08327674157670475],
    [0.0, 0.2334925365383547, 0.1019301198172404],
    [0.0, 0.2491470458134029, 0.1181945319615184],
    [0.0, 0.0, 0.1316886384491766],
    [0.0, 0.0, 0.1420961093183821],
    [0.0, 0.0, 0.1491729864726037],
    [0.0, 0.0, 0.1527533871307259],
];

/// Bivariate standard normal distribution
#[derive(Debug, Default, Clone, Copy)]
pub struct BvNormal;

impl BvNormal {
    /// Create a new bivariate normal distribution.
    pub fn new() -> Self {
        BvNormal
    }

    /// Returns P(X ≤ x, Y ≤ y) under N₂(0,0,1,1,ρ).
    pub fn cdf(&self, x: f64, y: f64, r: f64) -> f64 {
        let h = -x;
        let mut k = -y;

        // pick which column & how many points
        let (ng, lg) = match r.abs() {
            a if a < 0.3 => (0, 3),
            a if a < 0.75 => (1, 6),
            _ => (2, 10),
        };

        let mut hk = h * k;
        let mut bvn = 0.0;

        // “Main” vs. “tail” region switch
        if r.abs() < 0.925 {
            // Main region
            // hs = (x^2 + y^2)/2
            let hs = 0.5 * (h.powi(2) + k.powi(2));
            let asr = r.asin();

            for i in 0..lg {
                let xi = X[i][ng];
                let wi = W[i][ng];
                // first node
                let sn = ((xi + 1.0) * asr * 0.5).sin();
                bvn += wi * ((sn.mul_add(h * k, -hs)) / (1.0 - sn * sn)).exp();
                // mirrored node
                let sn = ((-xi + 1.0) * asr * 0.5).sin();
                bvn += wi * ((sn.mul_add(h * k, -hs)) / (1.0 - sn * sn)).exp();
            }

            // complete with univariate tails
            bvn * asr / (2.0 * TWOPI) + STANDARD_NORMAL.cdf(-h) * STANDARD_NORMAL.cdf(-k)
        } else {
            // Tail region
            if r < 0.0 {
                k = -k;
                hk = -hk;
            }

            if r.abs() < 1.0 {
                let asq = (1.0 - r) * (1.0 + r); // as is reserved keyword
                let a = asq.sqrt();
                let bs = (h - k).powi(2);
                let c = (4.0 - hk) / 8.0;
                let d = (12.0 - hk) / 16.0;

                // first piece
                bvn += a
                    * (-(bs / asq + hk) / 2.0).exp()
                    * (1.0 - c * (bs - asq) * (1.0 - d * bs / 5.0) / 3.0 + c * d * asq * asq / 5.0);

                // correction if not too extreme
                if hk > -160.0 {
                    let b = bs.sqrt();
                    bvn -= (-hk / 2.0).exp()
                        * TWOPI.sqrt()
                        * STANDARD_NORMAL.cdf(-b / a)
                        * b
                        * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
                }

                // the double‐integral correction
                let a = 0.5 * a;
                for i in 0..lg {
                    let xi = X[i][ng];
                    let wi = W[i][ng];

                    // part 1
                    let xs = (a * (xi + 1.0)).powi(2);
                    let rs = (1.0 - xs).sqrt();
                    bvn += a
                        * wi
                        * ((-(bs / (2.0 * xs)) - hk / (1.0 + rs)).exp() / rs
                            - (-(bs / xs + hk) / 2.0).exp() * (1.0 + c * xs * (1.0 + d * xs)));

                    // part 2
                    let xs = asq * (-xi + 1.0).powi(2) / 4.0;
                    let rs = (1.0 - xs).sqrt();
                    bvn += a
                        * wi
                        * (-(bs / xs + hk) / 2.0).exp()
                        * ((-hk * (1.0 - rs) / (2.0 * (1.0 + rs))).exp() / rs
                            - (1.0 + c * xs * (1.0 + d * xs)));
                }

                bvn /= -TWOPI;
            }

            // final adjustment for sign of ρ
            if r > 0.0 {
                bvn + STANDARD_NORMAL.cdf(-h.max(k))
            } else {
                -bvn + (STANDARD_NORMAL.cdf(-h) - STANDARD_NORMAL.cdf(-k)).max(0.0)
            }
        }
    }
}
