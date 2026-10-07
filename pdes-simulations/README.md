# PDE Simulations

Numerical companion code for the paper *A biharmonic equation with
discontinuous nonlinearities* (E. Arias, M. Calahorrano, A. Castro —
Electronic Journal of Differential Equations, Vol. 2024, No. 15).

For the mathematical content, read the paper:

- Abstract: https://ejde.math.txstate.edu/Volumes/2024/15/abstr.html
- Portfolio page: https://matcraft94.github.io/publications/biharmonic-ejde-2024/

## Contents

| Notebook | Content |
|---|---|
| `biharmonic_nonlinear_discontinuity.ipynb` | Mixed Ciarlet–Raviart FEM (σ = −Δu, P1×P1) for the biharmonic problem with discontinuous nonlinearities — 2026-10-07 rewrite |
| `test_fenics.ipynb` | Environment/solver smoke test |

## What the notebook verifies (2026-10-07 rewrite, executed in `docker.io/dolfinx/lab:stable`)

- **Method**: mixed Ciarlet–Raviart formulation — the original P1
  `grad(grad(u))` form was degenerate (grad grad of P1 vanishes a.e.; its
  "convergence in 2 iterations" was an artifact of a null matrix). Picard
  iteration with damping (θ=0.5), tanh-regularized discontinuity
  (ε=1e-4), LU/MUMPS saddle-point solve.
- **Manufactured solution** u=(1−r²)² (exact clamped solution of Δ²u=64):
  L2 error 1.684e-3 → 2.526e-4 and H1 error 1.072e-2 → 1.650e-3 from
  n=16→64 (observed orders ≈1.3–1.4, O(h)-like on a polygonal boundary);
  σ L2 error 1.87e-1 → 1.28e-1. Linear residuals 1e-13–1e-12.
- **Paper's literal case** q(u)=u³−u, a=0.5: Picard converges in 41
  iterations to **u≡0** — consistent with the theory: q(a)=−0.375<0
  violates the spectral condition (paper eq. 3.11), so the trivial
  solution is the expected outcome.
- **Non-trivial case** q(u)=150u−u³ (q(a)/a=149.75>λ₁≈104.4): 53 Picard
  iterations, umax=9.032, free-boundary fraction 0.718, fixed-point
  residual ‖·‖∞=7.96e-10, linear residual 1.04e-12.
- Documented deviations from the paper's parameters (coefficient 0.1 vs
  1.0, q variants, initial guess) with reasons, plus the non-triviality
  threshold c₁ empirically between 60 and 120 — consistent with λ₁ of the
  clamped disk.

## Reproduce

Run inside the official FEniCS container (no host install):

```bash
podman run --rm -v "$PWD":/home -w /home docker.io/dolfinx/lab:stable \
  jupyter nbconvert --to notebook --execute --inplace \
  biharmonic_nonlinear_discontinuity.ipynb --ExecutePreprocessor.timeout=2400
```
