# Neural PDEs Solver

A notebook-based research sandbox for physics-informed networks and neural
ODEs: five self-contained PyTorch/Pyro notebooks covering forward and inverse
differential-equation problems. Successes and failures are reported side by
side, with every number taken from actual training logs.

Full narrative: [case study](https://matcraft94.github.io/case-studies/neural-pdes/)

## Notebooks and verified results

| Notebook | Problem | Verified outcome |
|---|---|---|
| `double-pendulum-pinn.ipynb` | PINN with domain decomposition (5 subdomains over t ∈ [0, 2]) | training loss **0.0244** logged at epoch 100 of 200 (final loss not printed); the comparison against the `solve_ivp` reference is **visual only** — no numerical error metric exists. Known data-handling bugs (2026-10-07 audit): the IC loss is applied at a random `t[0]` because the DataLoader shuffles batches, and the energy-variation penalty differences unordered (shuffled) points, so neither term measures what it claims |
| `inverse-pde-nllsq.ipynb` | Inverse parameter estimation for a Poisson problem | NLLSQ reaches α = **0.9645** at epoch 900 of 1000 (still descending; ≈ **0.963** at the final step, ~3.7% error — no stopping criterion, "recovers" is generous). VarPro diverges to α ≈ −0.79 (**178%** error) — but this is an **implementation artifact** (the closed-form update ignores the data and α is never a trainable parameter), not a fair verdict on the method |
| `RDA-DN-NA.ipynb` | Oregonator (BZ) regression surrogate | train plateau **0.078**, held-out test **0.346** — but the MSE is computed on an `[N,1]`×`[N]` broadcast in **both** train and test cells, so each value is a pessimistic upper bound (`MSE_broadcast = MSE_real + 2·Cov`). The code solves a temporal **ODE**; the diffusion/advection coefficients (`Du, Dv, a`) are defined and never used. No seed or weights are saved, so the numbers are not reproducible as-is |
| `SD-ODEs-Neural.ipynb` | Neural ODE classification benchmarks (two moons, concentric circles, ANODE) | re-run **2026-10-07** (RTX 5070 Ti, seed 42, metrics persisted via an added `FINAL_METRICS` print): two moons, plain NODE (200 ep) — test acc **1.0**, test loss **2.18e-3**; concentric circles, plain NODE (300 ep) — test acc **0.97** but test loss **0.0715**: comparable accuracy yet ~4 orders of magnitude worse loss than the ANODE, the quantitative fingerprint of the documented topological "cheat" (stretching the plane instead of separating the annuli, cf. Dupont et al. 2019); concentric circles, ANODE with 1 augmented dimension (100 ep) — test acc **1.0**, test loss **5.69e-6** |
| `COVID.ipynb` | SIR parameter inference with Pyro SVI | **did not converge**: loss flat at ≈1.597e7, β = γ = initialization — reported as a failure |

## Stack

PyTorch, torchdiffeq (TorchDyn as reference), Pyro, NumPy/SciPy; `data/`
holds the COVID-19 daily cases CSV.

## Reproduce

Each notebook is self-contained; run top to bottom in Jupyter with a Python
3.10+ environment (`pip install torch pyro-ppl torchdiffeq`).
