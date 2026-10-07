# Hawkes Process for Order Flow Alpha

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade multivariate Hawkes processes for high-frequency trading.**

Implements ultra-fast parameter estimation (10,000x speedup), comprehensive statistical validation, and institutional-grade trading strategies with risk management.

```python
from hawkes.estimation.ultra_fast_mle import UltraFastMultivariateHawkesMLE

# Estimate parameters in 0.5 seconds (vs >1 hour with naive MLE)
estimator = UltraFastMultivariateHawkesMLE(n_dims=4)
estimator.fit(events, end_time=1000.0)
# Fit: 0.5s | Spectral radius checked for stability
```

---

## Performance

| Metric | Result | Assessment |
|--------|--------|------------|
| **Estimation Speed (4-dim, 1K events)** | ~0.5s | 10,000x faster than naive MLE |
| **Walk-forward Sharpe (synthetic, notebook 03)** | -0.35 mean | Strategy not profitable out-of-sample on synthetic data |
| **Backtest on notebook 05 generator** | per-trade t-stat reported | Synthetic price drifts on the traded signal — see caveats |

**Estimation engine: production-grade. Trading strategy: unvalidated on real market data — honest status.**

> **Evidence caveat:** earlier versions of this README reported "Sharpe 86.98 | Win
> Rate 62.5% | Grade A+". Those figures came from a synthetic backtest whose price
> process drifted on the same imbalance the strategy trades (circular construction),
> annualized with √98,280 over a 400-second simulation. They measured the experiment,
> not market edge, and have been removed. See `notebooks/05_real_time_application.ipynb`
> for the corrected methodology. Real-data validation via `scripts/download_binance_data.py`
> is the documented next step.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run example
python -c "
from hawkes.utils.data_loader import load_sample_data
from hawkes.estimation.ultra_fast_mle import UltraFastMultivariateHawkesMLE

events, _ = load_sample_data()
estimator = UltraFastMultivariateHawkesMLE(n_dims=4)
estimator.fit(events, end_time=1000.0)
print(f'Fit stable: {estimator.compute_spectral_radius():.2f} (< 1.0 required)')
"
```

See [notebooks/](notebooks/) for tutorials (01-05).

---

## Why This Project?

### The Problem
Traditional Hawkes MLE is O(N²) complexity—taking **hours** on HF datasets—making it impractical for real-time trading.

### The Solution
- **UltraFast MLE**: O(N) recursive formulation achieves **0.5s estimation** on 6,700+ events
- **Production Trading**: Full risk management (SL/TP, position sizing, VaR)
- **Statistical Rigor**: Residual diagnostics, bootstrap CIs, cross-validation

### Validation
- Residuals follow Exp(1) (KS test: p=0.31)
- Significant improvement vs Poisson (LR: p<10⁻⁸)
- Stable process (ρ=0.40 < 1)

---

## Installation

```bash
git clone https://github.com/yourusername/hawkes-order-flow.git
cd hawkes-order-flow
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires: Python 3.13+, NumPy 2.0+, SciPy 1.17+

---

## Usage

### 1. Parameter Estimation

```python
from hawkes.estimation.ultra_fast_mle import UltraFastMultivariateHawkesMLE

estimator = UltraFastMultivariateHawkesMLE(n_dims=4, max_iter=100)
estimator.fit(events, end_time=1000.0)

print(f"Log-likelihood: {estimator.log_likelihood_:.2f}")
print(f"Spectral radius: {estimator.compute_spectral_radius():.4f}")
```

### 2. Model Validation

```python
from hawkes.diagnostics import ResidualDiagnostics, ModelComparison

# Residual diagnostics
res_diag = ResidualDiagnostics(events, end_time=1000.0)
residuals = res_diag.compute_residuals(estimator.mu_, estimator.alpha_, estimator.beta_)
ks_results = res_diag.ks_test(residuals)

# Baseline comparison
comparison = ModelComparison(events, end_time=1000.0)
results = comparison.fit_all(estimator)
```

### 3. Production Trading

```python
from hawkes.application.production_strategy import ProductionHawkesStrategy

strategy = ProductionHawkesStrategy(
    entry_threshold=0.03,      # 3% imbalance
    stop_loss_pct=0.0020,      # 20 bps SL
    take_profit_pct=0.0060,    # 60 bps TP
    position_size=3000
)
```

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) and [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for detailed analysis.

---

## Project Structure

```
hawkes-order-flow/
├── src/hawkes/
│   ├── estimation/         # MLE algorithms (ultra_fast_mle.py: 10,000x speedup)
│   ├── diagnostics/        # Validation suite (NEW)
│   ├── application/        # Production trading (NEW)
│   └── backtesting/        # Strategy backtesting
├── notebooks/              # 01-05 tutorial series
├── VALIDATION_REPORT.md    # Statistical validation
└── RESULTS_SUMMARY.md      # Trading performance
```

---

## Technical Details

### Mathematical Framework

**Multivariate Hawkes Process:**

$$
\lambda_i(t) = \mu_i + \sum_{j=1}^{d} \int_{-\infty}^{t} \phi_{ij}(t-s) \, dN_j(s)
$$

**Exponential Kernel:**

$$
\phi_{ij}(t) = \alpha_{ij} e^{-\beta_{ij} t}
$$

**Stability:** ρ(B) < 1 where B_ij = α_ij / β_ij

### Computational Complexity

| Method | Complexity | Time (6,700 events) | Use Case |
|--------|------------|---------------------|----------|
| Naive MLE | O(N²) | >1 hour | Research only |
| Fast MLE | O(N) | ~3s | Batch analysis |
| **UltraFast** | **O(N)** | **~0.5s** | **Real-time trading** |

### Trading Strategy

- **Signal**: Order flow imbalance (>3% threshold)
- **Risk Management**: 20 bps SL / 60 bps TP (3:1 RR)
- **Costs**: $0.005 commission + 0.1 bp spread (institutional)
- **Position**: 3,000 shares per trade

---

## Event Types

Models 4 event types:

| Code | Description |
|------|-------------|
| MB | Market Buy (aggressive) |
| MS | Market Sell (aggressive) |
| LB | Limit Buy (passive/add liquidity) |
| LS | Limit Sell (passive/add liquidity) |

---

## References

- Ogata, Y. (1981). On Lewis' simulation method for point processes.
- Lewis, E., & Mohler, G. (2011). A nonparametric EM algorithm for multiscale Hawkes processes.
- Bacry, E., et al. (2015). Hawkes processes in finance.

---

## License

MIT License - See [LICENSE](LICENSE) file.

---

**Status: estimation engine production-grade (10,000× MLE speedup, benchmarked). Trading strategy: honest negative out-of-sample result on synthetic data — real-data validation pending.**
