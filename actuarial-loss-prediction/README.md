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

## Files and reproduce

`trabajo_final_E_ARIAS.Rmd` is the source; `trabajo_final_E_ARIAS.html` is the
rendered output with all tables. Knit the Rmd in RStudio, or open the HTML for
the full report. `data/` holds the claims dataset.
