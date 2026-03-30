# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Rust workspace containing coursework for financial engineering graduate courses. Each course is a separate crate:

- `fe621` — Financial Engineering (options pricing, derivatives, stochastic processes)
- `fe630` — Fixed income / portfolio topics
- `fe635` — Probability/statistics topics
- `fe680` — Interest rate models / advanced topics

## Commands

```bash
# Build a specific crate
cargo build -p fe680

# Check without building
cargo check -p fe680

# Run a specific crate
cargo run -p fe680

# Run with release optimizations
cargo run -p fe680 --release
```

There are no tests. Running `cargo run -p <crate>` executes whatever is uncommented in that crate's `main.rs`.

## Architecture

### Rust Workspace

All four crates share a workspace `Cargo.toml` at the root. `polars` is a shared workspace dependency. Each crate is independently runnable via `cargo run -p <name>`.

### Code Organization Pattern

Each crate follows the same structure:
- `src/lib.rs` — re-exports assignment modules; has `#![allow(non_snake_case)]` for math notation
- `src/main.rs` — calls specific problem functions; most are commented out
- `src/<module>/mod.rs` — re-exports problem submodules (e.g., `assn1`, `fin`)
- `src/<module>/prob1.rs`, `prob2.rs`, etc. — individual problem implementations

**To run a specific problem:** uncomment the relevant function call in `main.rs` and run `cargo run -p <crate>`.

The `fe680` crate uses `src/fin/` for the final exam (instead of `assn*`), alongside `src/assn*/` for regular assignments. Only the currently active module is exported from `lib.rs`.

### Python / Jupyter (fe621 only)

`fe621/` also contains Python Jupyter notebooks (managed with `uv`) for data collection and analysis alongside the Rust implementations:
- `data.py` / `data.ipynb` — fetches options/historical data via `yfinance`
- `assn*.ipynb`, `part*.ipynb` — notebook-based assignment write-ups
- `fe621/data/` — CSV files with SPX options data

Python dependencies are managed with `uv` via `fe621/pyproject.toml`.

### Key Dependencies by Crate

| Crate | Notable deps |
|-------|-------------|
| fe621 | `polars`, `nalgebra`, `statrs`, `burn` (ML), `plotly`, `plotters`, `yahoo_finance_api`, `rayon` |
| fe630 | `polars`, `nalgebra`, `argmin` (optimization), `plotters`, `yahoo_finance_api` |
| fe635 | `statrs` |
| fe680 | `polars`, `nalgebra`, `argmin`, `plotters`, `statrs`, `rand`/`rand_distr`, `rayon` |
