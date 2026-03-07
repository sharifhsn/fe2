# fe2

Rust implementations of financial engineering models across four graduate courses. Each course is a separate crate in a Cargo workspace.

## Courses

### fe621 — Financial Engineering (Options & Derivatives)

**Pricing models:**
- Black-Scholes (European call/put)
- Implied volatility extraction via bisection and secant methods
- Trigeorgis additive binomial tree (European and American options)
- Finite difference methods: Explicit (EFD), Implicit (IFD), Crank-Nicholson

**Monte Carlo:**
- Correlated multi-asset basket simulation using Cholesky decomposition
- European basket option pricing
- Exotic barrier-based basket option pricing

**Data pipeline:** Python notebooks (`data.ipynb`, `data.py`) fetch live options data via `yfinance`; Rust code prices against that data.

---

### fe630 — Portfolio Theory & Fixed Income

- Expected utility theory (log utility, power utility) with certainty equivalents and risk premia
- Mean-variance portfolio optimization
- Yield curve bootstrapping from mixed rate inputs (cash, forwards, swaps) → forward, discount, zero, and par curves
- DV01, modified duration, and convexity

---

### fe635 — Probability & Statistics

- Black-Scholes
- Binary option pricing via static replication

---

### fe680 — Interest Rate & Credit Models

**Interest rate:**
- Cubic spline yield curve interpolation with bond pricing, duration, and convexity
- Short-rate models: Vasicek, Cox-Ingersoll-Ross (CIR), Hull-White
- Yield curve bootstrapping (forward and discount curve)
- Convexity-adjusted swap derivatives with quanto adjustment

**Credit:**
- Credit Default Swap (CDS) pricing; hazard rate calibration via bisection
- CDO tranche pricing with Binomial loss distribution
- Gaussian one-factor latent variable model for portfolio loss distribution
- Gaussian two-factor model with Monte Carlo simulation of default times

---

## Running

Uncomment the desired function call in the crate's `main.rs`, then:

```bash
cargo run -p fe680          # run a specific crate
cargo run -p fe680 --release  # with optimizations (recommended for Monte Carlo)
cargo check -p fe680        # type-check without building
```
