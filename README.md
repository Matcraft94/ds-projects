# Data Science & Quantitative Research Portfolio

** Data Scientist | Quantitative Analyst | Mathematical Engineer**

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Production-ready implementations of statistical models, machine learning systems, and quantitative trading strategies. Focus on high-performance computing, rigorous validation, and institutional-grade code quality.

---

## Featured Projects

### [Hawkes Order Flow Alpha](./hawkes-order-flow/)

**High-Frequency Trading with Hawkes Processes**

Production-grade multivariate Hawkes process implementation for order flow prediction.


| Metric           | Result    | Assessment                       |
| ---------------- | --------- | -------------------------------- |
| **Estimation Speed** | ~0.5s (4-dim) | 10,000× vs naive MLE (benchmarked) |
| **Walk-forward Sharpe** | -0.35 (mean) | Negative out-of-sample — reported honestly |

- **Key contribution:** O(N) recursive MLE (0.5s vs >1h), statistical validation suite, real-time pipeline architecture
- **Honest caveat:** earlier "Sharpe 86.98" claims came from a circular synthetic backtest; removed. See project README.

[→ View Details](./hawkes-order-flow/)

---

### [Neural PDEs Solver](./neural-pdes-solver/)

**Physics-Informed Neural Networks for Differential Equations**

PINN implementations for forward and inverse differential-equation problems, with documented failures alongside the successes.

- Double pendulum PINN with domain decomposition over t ∈ [0, 2], fitted to a logged loss of 0.024 against the `solve_ivp` reference, with an energy-variation penalty in the loss
- NLLSQ inverse parameter recovery for a Poisson problem (α ≈ 0.965 vs true 1.0); VarPro and Pyro SVI failures documented with their logs
- Oregonator (BZ) regression surrogate with an explicit train/test gap (0.078 / 0.346, broadcast-MSE caveat noted)

[→ View Details](./neural-pdes-solver/)

---

### [Actuarial Loss Prediction](./actuarial-loss-prediction/)

**XGBoost-Based Claims Prediction System**

Automated actuarial loss prediction with advanced feature engineering and text processing.

- XGBoost with hyperparameter optimization and GPU acceleration
- NLP processing for claim descriptions
- Segment-wise analysis and pricing optimization
- End-to-end ML pipeline (test RMSE 25,034 USD, MAE 7,258 on the held-out 20%)

[→ View Details](./actuarial-loss-prediction/)

---

### [Market Risk Analysis](./market-risk-analysis/)

**LSTM next-bar prediction with walk-forward validation**

- 2-layer LSTM on 94,858 one-minute bars; walk-forward TimeSeriesSplit (3 folds) + untouched 20% chronological holdout
- Audited twice while documenting: seven methodological defects found and fixed, including an RSI NaN bug whose `dropna` silently deleted 21% of bars and biased every result (commit history has the full trail)
- Honest outcome: no directional edge at the 1-minute horizon — holdout backtest -0.20% vs a market that returned -0.18% close-to-close

[→ View Details](./market-risk-analysis/)

---

### [Academic Performance Prediction](./academic-performance-prediction/)

**Student Dropout Early Warning System**

Machine learning approach for early detection of at-risk students using LightGBM.

- LightGBM with GPU acceleration
- Comprehensive feature engineering (socioeconomic factors)
- 88% accuracy in dropout prediction
- Interactive performance dashboards

[→ View Details](./academic-performance-prediction/)

---

## Technical Expertise

### Core Competencies

- **Statistical Modeling:** Hawkes processes, Time-series analysis, Survival analysis
- **Machine Learning:** Gradient boosting, Neural networks, Bayesian methods
- **Quantitative Finance:** High-frequency trading, Risk management, Market microstructure
- **Scientific Computing:** PDE solvers, Numerical optimization, High-performance computing

### Technology Stack


| Category          | Tools                                                |
| ----------------- | ---------------------------------------------------- |
| **Languages**     | Python, R, MATLAB, SQL                               |
| **ML/DL**         | PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM |
| **Scientific**    | NumPy, SciPy, Pandas, Numba, Cython, FEniCS          |
| **Data**          | PostgreSQL, Neo4j, MongoDB                           |
| **MLOps**         | MLflow, Docker, GitHub Actions                       |
| **Visualization** | Matplotlib, Seaborn, Plotly, Jupyter                 |

---

## Repository Structure

```
ds-projects/
├── hawkes-order-flow/          # HFT with Hawkes processes (NEW)
├── neural-pdes-solver/         # Physics-informed neural networks
├── actuarial-loss-prediction/  # XGBoost claims prediction
├── market-risk-analysis/       # LSTM risk assessment
├── academic-performance-prediction/  # Student dropout prediction
├── pdes-simulations/           # FEM numerical solutions
└── README.md
```

---

## Research Publications

- **A biharmonic equation with discontinuous nonlinearities**
  *Eduardo Arias, Marco Calahorrano, Alfonso Castro*
  Electronic Journal of Differential Equations, 2024

---

## Connect

- **LinkedIn:** [linkedin.com/in/eduardo-arias-3e0](https://www.linkedin.com/in/eduardo-arias-3e0/)
- **Email:** mat.eduardo.arias@outlook.com
- **Location:** Ecuador

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*All projects include comprehensive documentation, statistical validation, and production-ready code.*
