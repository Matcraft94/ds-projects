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

## Model-family comparison (2026-10-07, `glm_gam_experiment.R`)

Same split/seed/protocol as above, adding classic actuarial families:
elastic-net GLM, Gamma(log) GLM, mgcv GAM (splines on continuous features),
GAMM (GAM + AccidentYear random effect), and a two-part **hurdle** XGB
(P(cost>50k) × Gamma severity + cheap-claims regressor).

| Model | RMSE | MAE | MAPE | R² | MAE Q5 |
|---|---|---|---|---|---|
| XGB log1p+TF-IDF (ref) | 23,851 | **6,094** | **60.1** | 0.286 | 25,394 |
| GLM elastic-net | 33,780 | 8,857 | 108 | −0.43 | 37,897 |
| GLM Gamma(log) | diverged | — | — | — | — |
| GAM (mgcv, Gamma log) | 23,388 | 8,097 | 129 | 0.314 | 28,639 |
| GAMM (+ year RE) | 23,388 | 8,097 | 129 | 0.314 | 28,639 |
| **Hurdle (two-part XGB)** | **22,800** | 6,650 | 77.5 | **0.348** | 25,856 |

Findings: the hurdle model posts the project's best RMSE/R², but the
single log-objective XGB still wins every relative-error metric (MAE −8%,
MAPE −22% vs hurdle) — the two-part structure trades tail accuracy for
slightly worse typical-claim fit. The GAM's splines beat the linear GLM by
a wide margin (RMSE 23.4k vs 33.8k) yet still lose to gradient boosting on
MAE; the year random effect contributes exactly nothing (years have
thousands of claims each, so the RE variance collapses — GAM ≡ GAMM to 6
decimals). The Gamma GLM diverges even winsorized — the classic actuarial
specification needs a tighter feature treatment than this pipeline gives
it. No family moves the needle on the expensive-claims quintile (~25k MAE
everywhere): with `InitialIncurredCalimsCost` dominating and the tail
intrinsically volatile, that segment stays open.

## Distribution study + targeted features (2026-10-07, `distribution_features_experiment.R`)

**Part A — where the error lives.** By Initial-cost decile (train): the
median Ultimate/Initial ratio swings 0.81–1.52, the *blow-up rate*
(ultimate > 3× initial) is **24% in the cheapest decile vs 6–7% in the
top**, and the within-decile SD of log-cost stays ≈0.75 everywhere.
Decomposing variance: **Initial alone explains 74.4% (deciles) / 76.3%
(percentiles) of the log-cost variance** — the within-percentile residual
SD is 0.742 in log space (total 1.524). Half the variance is
irreducible given the information available: the expensive-claims error
floor is structural, not a modeling failure.

**Part B — features aimed at that structure:** target-encoded stems and
bigrams (the original analysis computed per-term severity but never used
it as a feature; smoothed K=50, train-only), Initial-percentile target
encoding (the full nonlinear Initial→cost curve), `log_initial`,
zero-Initial flag, `log_initial×log(delay)` interaction, description
word count.

| Arm | RMSE | MAE | MAPE | R² | RMSLE |
|---|---|---|---|---|---|
| REF (TF-IDF, prior best config) | 23,851 | 6,094 | 60.1 | 0.286 | 0.724 |
| FE1 TF-IDF + new | 23,169 | 6,124 | 48.6 | 0.327 | 0.632 |
| **FE2 new only (no TF-IDF)** | 22,816 | **5,658** | **33.6** | 0.347 | **0.563** |
| FE3 hurdle + all | **22,540** | 6,612 | 59.7 | **0.363** | 0.649 |

Findings: (1) **two target-encoded text columns replace 300 TF-IDF
columns and beat them** — FE2 wins every relative-error metric (MAPE
−44%, RMSLE −22% vs REF; MAE −22% vs the published original) with a
smaller, more interpretable feature set; (2) the hurdle variant is again
best on RMSE/R²; (3) Q1–Q4 quintile MAE drops further (Q1 457→128) and
even Q5 edges down (25,394→24,603) — but Part A shows ~25k MAE in the
expensive segment sits near the information ceiling of this dataset.

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
