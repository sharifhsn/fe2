pub struct SwapDerivative {
    /// principal
    pub L: usize,
    /// frequency (e.g. 0.5 for semiannual)
    pub m: f64,
    /// time in the future when payoff occurs
    pub T: usize,
    /// period of time the swap rate is quoted for
    pub Ts: usize,
    /// risk-free rate for USD
    pub rfd: f64,
    /// risk-free rate for JPY
    pub rfy: f64,
    /// swap yield curve USD (assumed flat) per annum
    pub rd: f64,
    /// swap yield curve JPY (assumed flat) per annum
    pub ry: f64,
    /// forward swap rate volatility
    pub sigv: f64,
    /// forward exchange rate volatility
    pub sigw: f64,
    /// correlation between exchange rate and USD interest rate
    pub rho: f64,
}

impl SwapDerivative {
    pub fn new(L: usize, m: f64, T: usize, Ts: usize, rfd: f64,  rfy: f64, rd: f64, ry: f64, sigv: f64, sigw: f64, rho: f64) -> Self {
        Self {
            L,
            m,
            T,
            Ts,
            rfd,
            rfy,
            rd,
            ry,
            sigv,
            sigw,
            rho,
        }
    }

    pub fn G(&self) -> f64 {
        let mut total = 0.0;
        let T = self.Ts as f64 / self.m;
        let y = self.rd * self.m;
        let ts = (1..T as usize).map(|x| x as f64 * self.m).collect::<Vec<_>>();
        for t in ts {
            total += y / (1.0 + y).powf(t)
        }
        total += (1.0 + y) / (1.0 + y).powf(T);
        total
    }

    pub fn Gp(&self) -> f64 {
        let mut total = 0.0;
        let T = self.Ts as f64 / self.m;
        let y = self.rd * self.m;
        let ts = (1..T as usize).map(|x| x as f64).collect::<Vec<_>>();
        for t in ts {
            total -= y * self.m * t / (1.0 + y).powf(t + 1.0)
        }
        total -= (1.0 + y) * self.m * T / (1.0 + y).powf(T + 1.0);
        total
    }
    pub fn Gpp(&self) -> f64 {
        let mut total = 0.0;
        let T = self.Ts as f64 / self.m;
        let y = self.rd * self.m;
        let ts = (1..T as usize).map(|x| x as f64).collect::<Vec<_>>();
        for t in ts {
            total += y * self.m.powi(2) * t * (t + 1.0) / (1.0 + y).powf(t + 2.0);
            // println!("numerator: {}", y * self.m.powi(2));
        }
        total += (1.0 + y) * self.m.powi(2) * T * (T + 1.0) / (1.0 + y).powf(T + 2.0);
        total
    }

    pub fn convexity_adjusted_yield(&self) -> f64 {
        let Gp = self.Gp();
        let Gpp = self.Gpp();
        self.rd + 0.5 * self.rd.powi(2) * self.sigv.powi(2) * self.T as f64 * Gpp / Gp.abs()
    }

    pub fn quanto_adjustment(&self) -> f64 {
        // assumes payoff is applied to principal in foreign currency
        // and denominated in foreign currency
        (-self.rho * self.sigv * self.sigw * self.T as f64).exp()
    }

    pub fn price(&self, is_foreign: bool) -> f64 {
        let rd = self.convexity_adjusted_yield();
        let quanto = if is_foreign {
            self.quanto_adjustment()
        } else {
            1.0
        };
        let rf = if is_foreign {
            self.rfy
        } else {
            self.rfd
        };
        self.L as f64 * rd * quanto / (1.0 + rf * self.m).powf(self.T as f64 / self.m)
    }
}

pub fn G_prime(y: f64) -> f64 {
    let mut total = 0.0;
    for i in 1..=6 {
        total -= 2.0 * i as f64 / (1.0 + y / 2.0).powf(i as f64 + 1.0);
    }
    total -= 300.0 / (1.0 + y / 2.0).powf(7.0);
    total
}

pub fn a() {
    let L = 1_000_000;
    let m = 0.5;
    let T = 2;
    let Ts = 4;
    let rd = 0.0428;
    let ry = 0.01338;
    let rfd = 0.042;
    let rfy = 0.012;
    let sigv = 0.15;
    let sigw = 0.1;
    let rho = 0.35;


    let sd = SwapDerivative::new(
        L,
        m,
        T,
        Ts,
        rfd,
        rfy,
        rd,
        ry,
        sigv,
        sigw,
        rho,
    );
    println!("G: {}", sd.G());
    println!("Gp: {}", sd.Gp());
    println!("Gpp: {}", sd.Gpp());
    println!("convexity_adjusted_yield: {}", sd.convexity_adjusted_yield());
    println!("price: {}", sd.price(false));
    println!("quanto_adjustment: {}", sd.quanto_adjustment());
    println!("yen price: {}", sd.price(true));
}
