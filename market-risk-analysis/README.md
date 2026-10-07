# Market Risk Analysis

LSTM next-bar close prediction on 1-minute crash500 bars with walk-forward
validation and a cost-aware backtest. The value of this project is the audit
trail, not the alpha: seven methodological defects were found and fixed while
documenting it, including an RSI NaN bug whose `dropna` silently deleted ~21%
of the bars and biased every result.

Full narrative with the defect table and interactive charts:
[case study](https://matcraft94.github.io/case-studies/market-risk/)

## Data

94,858 one-minute bars of crash500 (2022-04-25 → 2022-06-30, 24/7 market),
94,789 rows × 13 features after engineering. 80/20 chronological split:
75,831 train/validation, 18,958 final hold-out.

## Verified results (re-run after all fixes, seed 42)

| Evaluation | Result |
|---|---|
| Fold validation losses (walk-forward, 3 folds) | 0.0070 / 0.9514 / 0.0491 (mean 0.336 ± 0.436) |
| Final hold-out loss | 4.55 (regime shift; model does not transfer) |
| Backtest total return (hold-out) | **-0.20%** vs market -0.18% close-to-close |
| Position changes | ~5 (signal is long ~100% of the hold-out) |
| Max drawdown | -5.0% (intraperiod fluctuation of the always-long position) |

Honest headline: corrected for the selection bias, the strategy is
buy-and-hold of a flat market — the model has no directional edge at the
1-minute horizon on this instrument.

## Reproduce

```bash
uv sync                 # pyproject.toml pins torch (CPU), pandas, scikit-learn, plotly
uv run python main.py   # ~15 min on CPU; seeds pinned to 42
```

Trained models (`model_fold_*.pth`, `best_model.pth` + `best_model_scaler.npz`)
and the latest run artifacts are committed.
