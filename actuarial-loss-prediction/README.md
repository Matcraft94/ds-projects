# Actuarial Loss Prediction

End-to-end R/tidymodels pipeline that predicts ultimate incurred claim cost
from tabular, temporal, and free-text claim data (~54k claims), with
hyperparameter tuning via Latin hypercube search and error analysis by value
segment.

Full narrative with caveats and the importance chart:
[case study](https://matcraft94.github.io/case-studies/actuarial-loss/)

## Verified results (from the rendered analysis document)

| Metric | Train | Test (held-out 20%) |
|---|---|---|
| RMSE | 22,216.72 | **25,033.93** USD |
| MAE | 6,744.16 | **7,257.61** USD |

- **Gain importance** (via `vip::vi()` — gain, not SHAP despite the source
  plot's subtitle): `InitialIncurredCalimsCost` dominates at **0.879**;
  engineered `WeeklyWagesPerHour` 0.024; text stem `CD_hand` is second
  overall at **0.032**, ahead of weekly wages (0.016) and age (0.010).
- Free-text `ClaimDescription` fields were lowercased, stop-word removed and
  stemmed (SnowballC); the top ~100 stems (min frequency 50) became stem-count
  features (`CD_` prefix). Bigrams were frequency-analyzed only.
- Defensive capping code for infinite target values exists; it is a no-op on
  this dataset (none found).

## Known flaws (disclosed)

- The 80/20 split was sampled *before* text-feature extraction and target
  imputation, both computed on combined data — test metrics mildly optimistic.
- Missing ultimate costs were filled with group means (biases target downward).
- The reported R² is squared correlation, not the coefficient of determination.
- **Environment-bound numbers:** a 2026-10-07 re-run (data re-downloaded from
  a public mirror, seeds unchanged) reproduces the pipeline end-to-end but
  not the exact figures (RMSE 29,032/26,355 vs 22,217/25,034; gain 0.838 vs
  0.879) — the published numbers belong to the original 2025 environment,
  preserved in the rendered `analysis.html`.

## Improved model (2026-10-07, `improved_model.R`)

A clean-protocol experiment in a single environment (same container, same
data, seed 42), three arms: **A** original design with audit fixes only
(seeded split first, no target imputation); **B** `log1p` target + TF-IDF
text features (200 stems + 100 bigrams, IDF from train only); **C** XGBoost
Tweedie objective + TF-IDF. All arms: mini-grid (eta × depth), internal
validation, refit, single untouched-test evaluation.

| Arm | RMSE | MAE | MAPE | R² | RMSLE |
|---|---|---|---|---|---|
| faithful re-run (reference) | 26,354 | 7,730 | 148.9 | 0.274 | — |
| A baseline-clean | 30,437 | 11,010 | 340 | −0.16 | 2.71 |
| **B log1p + TF-IDF** | 23,476 | **6,050** | **59.2** | 0.309 | **0.719** |
| **C Tweedie + TF-IDF** | **22,871** | 6,826 | 88.1 | **0.344** | 0.799 |

Findings, all machine-verified: (1) the gain comes from the objective
transform + text representation, **not** from removing the leaky split —
arm A alone is *worse* than the faithful re-run (its own mini-grid is
weaker than the original Latin-hypercube tuning); (2) B wins every
relative-error metric (MAE −22% vs same-env baseline, −17% vs the
published original) because the squared-dollar objective was ignoring the
cheap-claims majority — quintile MAE for Q1–Q4 drops ~3–5× (e.g. Q4:
7,479 → 2,164); (3) C is the best RMSE/R² choice (RMSE −13% same-env,
−8.6% vs published); (4) the >Q5 expensive-claims segment remains the
dominant error (~25k MAE) in every arm — reserve accuracy for
high-value claims stays an open problem; (5) `InitialIncurredCalimsCost`
is the top feature in all arms, as in the original.

## Files and reproduce

- `analysis.Rmd` — source (R, tidymodels); `analysis.html` — rendered report
  with all tables and plots (the verified-numbers source of record).
- `Data/actuarial_loss/train.csv` — **not included**: Kaggle competition
  *Actuarial Loss Prediction* (workers' compensation). Download it and place
  it at that path, then `Rscript -e 'rmarkdown::render("analysis.Rmd")'`.
  Expected shape: ~54,000 rows; key columns include `ClaimNumber`,
  `DateTimeOfAccident`, `ClaimDescription`, `InitialIncurredCalimsCost`,
  `UltimateIncurredClaimCost` (target).

## Reproducibility note (2026-10-07 re-run)

The original CSV was lost, so the data was re-downloaded from a public
mirror of the competition (`github.com/riwajpokhrel/Actuarial_loss_prediction`)
— 54,000 rows, header identical including the upstream `CalimsCost` typo.
The pipeline then runs end-to-end (R 4.6.1, current tidymodels/xgboost,
containerized), but the metrics **shift across environments**: train/test
RMSE 29,031.85/26,354.52 and gain importance 0.838 (vs the original
22,216.72/25,033.93 and 0.879), despite identical seeds (123/345) —
package-version drift and/or row ordering in the mirror. Conclusion: the
numbers above are **environment-bound to the original 2025 run** whose
rendered output (`analysis.html`) remains the source of record; the re-run
output (`analysis_verify.html`, untracked) documents this check.
