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


| Metric           | Result  | Assessment    |
| ---------------- | ------- | ------------- |
| **Sharpe Ratio** | 86.98   | Exceptional   |
| **Win Rate**     | 62.5%   | Strong        |
| **Speedup**      | 10,000x | UltraFast MLE |

- **Key Innovations:** O(N) recursive MLE (0.5s vs >1h), real-time production trading, comprehensive statistical validation
- **Technologies:** Numba, Cython, Time-series CV, Bootstrap inference
- **Grade:** A+ | Production Ready

[→ View Details](./hawkes-order-flow/)

---

### [Neural PDEs Solver](./neural-pdes-solver/)

**Physics-Informed Neural Networks for Differential Equations**

PINN implementation for solving forward and inverse PDE problems with energy conservation guarantees.

- Double pendulum system with domain decomposition (MSE < 1e-3)
- NLLSQ and VarPro methods for inverse problems
- GPU memory optimization and adaptive activation functions
- Residual network architecture with energy conservation losses

[→ View Details](./neural-pdes-solver/)

---

### [Actuarial Loss Prediction](./actuarial-loss-prediction/)

**XGBoost-Based Claims Prediction System**

Automated actuarial loss prediction with advanced feature engineering and text processing.

- XGBoost with hyperparameter optimization and GPU acceleration
- NLP processing for claim descriptions
- Segment-wise analysis and pricing optimization
- End-to-end ML pipeline (RMSE: 23,669)

[→ View Details](./actuarial-loss-prediction/)

---

### [Market Risk Analysis](./market-risk-analysis/)

**LSTM-Based Risk Assessment Platform**

Market risk evaluation using deep learning and statistical arbitrage detection.

- LSTM networks for market crash prediction
- Robust preprocessing pipeline (noise reduction: 8.7%)
- Trading simulation with risk metrics (Sharpe, Drawdown)
- Early stopping convergence in 10 epochs

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
├── academic-performance/       # Student dropout prediction
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
