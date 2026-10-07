# Neural PDEs Solver

A notebook-based research sandbox for physics-informed networks and neural
ODEs: five self-contained PyTorch/Pyro notebooks covering forward and inverse
differential-equation problems. Successes and failures are reported side by
side, with every number taken from actual training logs.

Full narrative: [case study](https://matcraft94.github.io/case-studies/neural-pdes/)

## Notebooks and verified results

| Notebook | Problem | Verified outcome |
|---|---|---|
| `double-pendulum-pinn.ipynb` | PINN with domain decomposition (5 subdomains over t ∈ [0, 2]) | logged loss **0.024** vs `solve_ivp` reference; energy-variation penalty in the loss |
| `inverse-pde-nllsq.ipynb` | Inverse parameter estimation for a Poisson problem | NLLSQ recovers α ≈ **0.965** (true 1.0); VarPro fails (178% error) — documented |
| `RDA-DN-NA.ipynb` | Oregonator (BZ) regression surrogate | train plateau **0.078**, held-out test **0.346** — MSE computed on an `[80,1]`×`[80]` broadcast (caveat in the case study) |
| `SD-ODEs-Neural.ipynb` | Neural ODE classification benchmarks (two moons, concentric circles, ANODE) | re-run **2026-10-07** (RTX 5070 Ti, seed 42, metrics persisted via an added `FINAL_METRICS` print): two moons, plain NODE (200 ep) — test acc **1.0**, test loss **2.18e-3**; concentric circles, plain NODE (300 ep) — test acc **0.97** but test loss **0.0715**: comparable accuracy yet ~4 orders of magnitude worse loss than the ANODE, the quantitative fingerprint of the documented topological "cheat" (stretching the plane instead of separating the annuli, cf. Dupont et al. 2019); concentric circles, ANODE with 1 augmented dimension (100 ep) — test acc **1.0**, test loss **5.69e-6** |
| `COVID.ipynb` | SIR parameter inference with Pyro SVI | **did not converge**: loss flat at ≈1.597e7, β = γ = initialization — reported as a failure |

## Stack

PyTorch, torchdiffeq (TorchDyn as reference), Pyro, NumPy/SciPy; `data/`
holds the COVID-19 daily cases CSV.

## Reproduce

Each notebook is self-contained; run top to bottom in Jupyter with a Python
3.10+ environment (`pip install torch pyro-ppl torchdiffeq`).
