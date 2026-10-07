# Neural PDEs Solver

A notebook-based research sandbox for physics-informed networks and neural
ODEs: five self-contained PyTorch/Pyro notebooks covering forward and inverse
differential-equation problems. Successes and failures are reported side by
side, with every number taken from actual training logs.

Full narrative: [case study](https://matcraft94.github.io/case-studies/neural-pdes/)

## Notebooks and verified results

| Notebook | Problem | Verified outcome |
|---|---|---|
| `double-pendulum-pinn.ipynb` | PINN with domain decomposition (5 subdomains over t ∈ [0, 2]) | fixed 2026-10-07 (IC loss now evaluated at t=0; energy penalty computed on time-sorted points) and re-run on GPU: final training loss **0.00198** (epoch 199 of 200; the old 0.0244 was an epoch-100 intermediate of the buggy run). New honest reference metrics vs `solve_ivp` (RK45, rtol=atol=1e-10): relative L2 error **0.86–0.97 per component** and energy drift ~36 J — the PINN fits the residuals yet does **not** reproduce the chaotic reference trajectory. Low training loss ≠ correct solution; both facts reported |
| `inverse-pde-nllsq.ipynb` | Inverse parameter estimation for a Poisson problem | re-implemented + re-run 2026-10-07: **the benchmark is α-unidentifiable by construction** (the manufactured source f(α) shares α with the coefficient, so the residual vanishes for every α at u = u_true). With a correct VarPro (Golub–Pereyra: exact head elimination + exact 1-D α step) the **solution** is recovered to machine precision (R² 1.0, MSE 3.2e-13, 9.4 s — 10× faster than NLLSQ) while α is not (−0.005, 100.5% error); NLLSQ initialized at the true α=1.0 drifts to 0.908 without plateau, and solution R² degrades with training (0.44→0.31). The old «NLLSQ recovers 0.965» was an init artifact; the old «VarPro fails 178%» was an implementation artifact — verdict corrected in-notebook |
| `RDA-DN-NA.ipynb` | Oregonator (BZ) regression surrogate | broadcast-MSE bug fixed 2026-10-07 (`.squeeze(-1)` in train and test): real MSE **1.28e-6** train / **1.48e-5** test (seed 42, weights saved to `RDA-DN-NA_model.pth`) — the previous 0.078/0.346 were the pessimistic `MSE + 2·Cov` broadcast bound; the model actually converges. Still a temporal ODE (the RDA coefficients are defined and unused) — documented in-notebook |
| `SD-ODEs-Neural.ipynb` | Neural ODE classification benchmarks (two moons, concentric circles, ANODE) | re-run **2026-10-07** (RTX 5070 Ti, seed 42, metrics persisted via an added `FINAL_METRICS` print): two moons, plain NODE (200 ep) — test acc **1.0**, test loss **2.18e-3**; concentric circles, plain NODE (300 ep) — test acc **0.97** but test loss **0.0715**: comparable accuracy yet ~4 orders of magnitude worse loss than the ANODE, the quantitative fingerprint of the documented topological "cheat" (stretching the plane instead of separating the annuli, cf. Dupont et al. 2019); concentric circles, ANODE with 1 augmented dimension (100 ep) — test acc **1.0**, test loss **5.69e-6** |
| `COVID.ipynb` | SIR parameter inference with Pyro SVI | **fixed and converged 2026-10-07**: root cause was **I(0)=0** — the first 38 CSV rows have zero infected, freezing the Euler dynamics and killing all gradients (the flat loss was not a scale problem alone). Trimming to days with cases (from 2020-02-29) + population-fraction normalization: **β=0.193, γ=0.119, R₀≈1.62**, ELBO −62% (29,476→11,187 over 2000 iters, monotone). Non-convergence documented as fixed failure→success |

## Stack

PyTorch, torchdiffeq (TorchDyn as reference), Pyro, NumPy/SciPy; `data/`
holds the COVID-19 daily cases CSV.

## Reproduce

Each notebook is self-contained; run top to bottom in Jupyter with a Python
3.10+ environment (`pip install torch pyro-ppl torchdiffeq`).
