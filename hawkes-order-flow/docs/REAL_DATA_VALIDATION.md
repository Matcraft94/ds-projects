# Real-Data Validation — Binance BTCUSDT

Date: 2026-10-06
Data: 73,375 aggregate trades, BTCUSDT, 2026-10-05 00:00–02:00 UTC
(via `scripts/download_binance_data.py` with corrected pagination)
Events: 39,330 aggressive buys / 34,045 aggressive sells
(aggTrades reveal only the taker side — bivariate model)

## What was validated

1. **Estimation engine on real data.** UltraFast bivariate MLE fit on
   39,815 training events (first 60%): spectral radius 0.2800 — a stable,
   self-exciting process, exactly the regime Hawkes models are designed
   for. The ~10,000× speedup claim is about this estimator, and it works
   on real market data.

2. **The trading signal on real prices.** Same parameters as the synthetic
   notebook 05 (3% entry threshold, 20bps SL / 60bps TP, 3s cooldown,
   2bps/side + 1bp spread costs, 10% notional), evaluated on a 1-second
   mark-to-market grid over the last 40% of the window (2,880 seconds).

## Results (test window, real prices)

| Metric | Value |
|--------|-------|
| Trades | 55 |
| Win rate | 0.0% |
| Net return | -0.27% |
| Max drawdown | 0.27% |
| Sharpe (annualized by **actual elapsed seconds**) | -14.60 |
| Per-trade P&L variance | ~0 (t-stat n/a) |

## Interpretation

The loss equals the transaction-cost drag almost exactly
(55 trades × 10% notional × 0.05% round-trip ≈ 0.275%). Every trade
closed at its stop-loss boundary distance equal to costs — the order-flow
imbalance signal carries **no predictive power** for next-second price
movement on this window. The impressive synthetic results (Sharpe 86.98,
62.5% win rate) were entirely produced by the simulated price drifting on
the signal itself; on real data that construction is absent and the
strategy is noise minus costs.

The negative Sharpe is methodology-sound (per-second mark-to-market
returns annualized with √(31,536,000 / 2,880)) but describes a single
48-minute window — illustrative of methodology, not an estimate of any
population property.

## Reproduce

```bash
python scripts/download_binance_data.py --symbol BTCUSDT \
    --start 2026-10-05 --end 2026-10-05
python scripts/validate_real_data.py --csv data/raw/BTCUSDT_..._trades.csv
```
