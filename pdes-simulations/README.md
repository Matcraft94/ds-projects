# PDE Simulations

Numerical companion code for the paper *A biharmonic equation with
discontinuous nonlinearities* (E. Arias, M. Calahorrano, A. Castro —
Electronic Journal of Differential Equations, Vol. 2024, No. 15).

This directory is intentionally code-only: the notebooks contain FEniCS
experiments behind the paper's results, with no narrative claims of their
own. For the mathematical content, read the paper:

- Abstract: https://ejde.math.txstate.edu/Volumes/2024/15/abstr.html
- Portfolio page: https://matcraft94.github.io/publications/biharmonic-ejde-2024/

## Contents

| Notebook | Content |
|---|---|
| `biharmonic_nonlinear_discontinuity.ipynb` | FEniCS finite-element experiments for the biharmonic problem with discontinuous nonlinearities |
| `test_fenics.ipynb` | Environment/solver smoke test |

## Reproduce

Requires a FEniCS/dolfin environment (e.g. `conda install -c conda-forge fenics`)
and Jupyter. Run top to bottom.
