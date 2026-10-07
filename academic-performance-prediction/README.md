# Academic Performance Prediction

Binary early-warning classifier for student dropout on the UCI *Predict
students' dropout and academic success* dataset (4,424 students, 34
attributes), with 13 engineered academic/socioeconomic features, random-search
hyperparameter optimization under 5-fold CV, and an ensemble of fold models.

Full narrative with caveats and charts:
[case study](https://matcraft94.github.io/case-studies/academic-performance/)

## Verified results (from the committed notebook outputs)

- Protocol: 80/20 split (`random_state=42`) → 3,539 train / 885 held-out
  students; all search and CV inside the training set.
- 5-fold CV validation precision 0.856 / 0.884 / 0.879 / 0.859 / 0.864 —
  notebook-printed mean **0.8683 ± 0.0111** (fold accuracy as displayed:
  0.86 / 0.88 / 0.88 / 0.86 / 0.86).
- Held-out test (885 students: 271 dropouts, 614 non-dropouts): **accuracy
  0.88**, precision 0.91, recall 0.92, F1 0.92 — dropout recall **0.79** is
  the operational caveat for an early-warning system.
- Random search best mean CV score 0.8706 (LightGBM `goss`, 300 trees).

## Known flaws (disclosed, not hidden)

- Label encoders were fitted before the split (mild leakage — a production
  version fits encoders on the training fold only).
- `select_features` / `remove_multicollinearity` helpers exist but were never
  called (dead code); the model trains on the full encoded feature set.
- Hyperparameter search scored by the same CV it reports — mild optimism.

## Reproduce

```bash
pip install lightgbm pandas scikit-learn plotly
jupyter notebook academic_performance_prediction.ipynb   # run top to bottom
```
