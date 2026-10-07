#!/usr/bin/env python3
"""Validate the Hawkes order-flow pipeline on real Binance trade data.

Unlike notebooks 03/05, the price series here is real market data — there is
no injected drift, so the strategy gets a fair test. Events are bivariate
(aggressive buy/sell): Binance aggTrades only reveal the taker side.

Methodology notes (deliberate, read before citing numbers):
- Order-flow imbalance: trailing 15s count imbalance of aggressive trades,
  evaluated on a 1-second grid.
- Entries: |imbalance| > 3%, direction = sign(imbalance), one position at a
  time, 3s cooldown (same parameters as notebook 05).
- Exits: stop-loss 20bps / take-profit 60bps checked against 1s close prices
  (conservative approximation — intra-second extremes are not observable on
  this grid), or time exit after 50s.
- Costs: 2bps per side (Binance VIP taker) + 1bp spread, per round trip.
- Position: 10% of equity notional, marked to market every second.
- Sharpe: annualized from per-second mark-to-market equity returns using the
  ACTUAL elapsed seconds of the test window (sqrt(31_536_000 / T)). This is
  the formula notebooks 03/05 got wrong; here it is applied to real time.
  Still a single-window estimate — treat as illustrative, not production PnL.

Usage:
    python scripts/validate_real_data.py [--csv data/raw/BTCUSDT_..._trades.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hawkes.estimation.ultra_fast_mle import UltraFastMultivariateHawkesMLE  # noqa: E402
from hawkes.utils.data_loader import trades_to_hawkes_events  # noqa: E402

# Strategy parameters (kept aligned with notebook 05 for comparability)
ENTRY_THRESHOLD = 0.03
STOP_LOSS_PCT = 0.0020
TAKE_PROFIT_PCT = 0.0060
MAX_HOLD_SECONDS = 50.0
COOLDOWN_SECONDS = 3.0
COST_PER_SIDE = 0.0002  # Binance VIP taker fee
SPREAD = 0.0001  # 1bp
POSITION_FRACTION = 0.10
SECONDS_PER_YEAR = 31_536_000.0


def load_events(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    events, meta = trades_to_hawkes_events(df)
    # Real aggTrades only expose the taker side: use bivariate (MB, MS).
    events2 = [events[0], events[1]]
    return df, events2, meta


def build_second_grid(df: pd.DataFrame, t0: float, t1: float):
    """1-second grid of last trade price over [t0, t1] (seconds from df start)."""
    n = int(t1 - t0) + 1
    grid_t = np.arange(n, dtype=float)  # grid_t[i] = t0 + i
    trade_t = ((df["time"] - df["time"].min()).dt.total_seconds()).values
    mask = (trade_t >= t0) & (trade_t <= t1)
    tt = trade_t[mask]
    pp = df["price"].values[mask]
    if len(tt) == 0:
        raise ValueError("no trades in test window")
    idx = np.searchsorted(tt, grid_t, side="right") - 1
    idx = np.clip(idx, 0, len(tt) - 1)
    close = pp[idx]
    close[: np.searchsorted(tt, t0, side="right")] = pp[0]
    return grid_t, close


def rolling_imbalance(event_times: list, grid_t: np.ndarray, window: float = 15.0):
    """Trailing-window aggressive buy/sell imbalance at each grid point."""
    lo = np.searchsorted(event_times[0], grid_t - window)
    hi = np.searchsorted(event_times[0], grid_t)
    lo_s = np.searchsorted(event_times[1], grid_t - window)
    hi_s = np.searchsorted(event_times[1], grid_t)
    n_buy = hi - lo
    n_sell = hi_s - lo_s
    total = n_buy + n_sell
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(total > 3, (n_buy - n_sell) / total, 0.0)
    return imb


def run_backtest(grid_t: np.ndarray, close: np.ndarray, imb: np.ndarray):
    equity = 100_000.0
    eq_curve = np.empty_like(close)
    position = 0
    entry_price = None
    entry_i = None
    last_trade_t = -np.inf
    trades = []

    for i in range(len(grid_t)):
        ret = 0.0 if i == 0 else close[i] / close[i - 1] - 1.0

        if position != 0:
            pnl_pct = (close[i] / entry_price - 1.0) * position
            holding = grid_t[i] - grid_t[entry_i]
            exit_reason = None
            if pnl_pct <= -STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                exit_reason = "TAKE_PROFIT"
            elif holding >= MAX_HOLD_SECONDS:
                exit_reason = "TIME_EXIT"
            if exit_reason:
                gross = pnl_pct
                cost = 2 * COST_PER_SIDE + SPREAD
                net = gross - cost
                equity *= 1 + POSITION_FRACTION * net
                trades.append({"pnl_pct": net, "win": net > 0, "exit": exit_reason})
                position = 0
                entry_price = None
                last_trade_t = grid_t[i]

        if (
            position == 0
            and grid_t[i] - last_trade_t >= COOLDOWN_SECONDS
            and abs(imb[i]) > ENTRY_THRESHOLD
        ):
            position = 1 if imb[i] > 0 else -1
            entry_price = close[i]
            entry_i = i

        eq_curve[i] = equity

    if position != 0:
        pnl_pct = (close[-1] / entry_price - 1.0) * position
        net = pnl_pct - (2 * COST_PER_SIDE + SPREAD)
        equity *= 1 + POSITION_FRACTION * net
        trades.append({"pnl_pct": net, "win": net > 0, "exit": "FINAL"})
        eq_curve[-1] = equity

    return eq_curve, trades


def report(eq_curve: np.ndarray, trades: list, test_seconds: float):
    pnls = np.array([t["pnl_pct"] for t in trades])
    n = len(pnls)
    print("=" * 70)
    print("REAL-DATA VALIDATION REPORT (Binance BTCUSDT)")
    print("=" * 70)
    print(f"\nTrade Statistics")
    print("-" * 70)
    print(f"  Total Trades:       {n:>12,d}")
    if n > 0:
        print(f"  Win Rate:           {np.mean(pnls > 0) * 100:>11.1f}%")
        gp = pnls[pnls > 0].sum()
        gl = -pnls[pnls < 0].sum()
        pf = gp / gl if gl > 0 else float("inf")
        print(f"  Profit Factor:      {pf:>12.2f}")
        if pnls.std() > 1e-12:
            t_stat = pnls.mean() / pnls.std() * np.sqrt(n)
            print(f"  Per-trade t-stat:   {t_stat:>12.2f}")
        else:
            print(f"  Per-trade t-stat:   {'n/a (zero variance)':>20}")

    rets = np.diff(eq_curve) / eq_curve[:-1]
    dur = test_seconds
    if rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(SECONDS_PER_YEAR / dur)
    else:
        sharpe = 0.0
    running_max = np.maximum.accumulate(eq_curve)
    max_dd = np.max((running_max - eq_curve) / running_max)

    print(f"\nRisk Metrics (per-second mark-to-market, actual elapsed time)")
    print("-" * 70)
    print(f"  Test window:        {dur:>11,.0f}s")
    print(f"  Sharpe (annualized by real time): {sharpe:>7.2f}")
    print(f"  Max Drawdown:       {max_dd * 100:>11.2f}%")
    print(f"  Final Equity:       ${eq_curve[-1]:>11,.2f}")
    print(f"  Net Return:         {(eq_curve[-1] / eq_curve[0] - 1) * 100:>11.2f}%")
    print("=" * 70)
    print("Caveat: single test window, minutes of data. Illustrative of")
    print("methodology, not evidence of production PnL.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Trades CSV from download_binance_data.py")
    parser.add_argument("--train-frac", type=float, default=0.6)
    args = parser.parse_args()

    df, events, meta = load_events(args.csv)
    duration = meta["duration_seconds"]
    split = duration * args.train_frac
    print(f"Loaded {len(df)} trades, {duration:.0f}s. Split at {split:.0f}s")

    train_events = [e[e < split] for e in events]
    print(f"Fitting bivariate Hawkes on {len(train_events[0])} MB / {len(train_events[1])} MS events...")
    estimator = UltraFastMultivariateHawkesMLE(n_dims=2, assume_independent=True, max_iter=100)
    estimator.fit(train_events, end_time=split)
    rho = estimator.compute_spectral_radius()
    print(f"Spectral radius: {rho:.4f} {'(stable)' if rho < 1 else '(UNSTABLE)'}")

    grid_t, close = build_second_grid(df, split, duration)
    imb = rolling_imbalance(events, grid_t)
    eq_curve, trades = run_backtest(grid_t, close, imb)
    report(eq_curve, trades, duration - split)


if __name__ == "__main__":
    main()
