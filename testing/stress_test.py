"""
stress_test.py — Walk-forward + Monte Carlo + Parameter sensitivity validation
Imports from backtest.py for core strategy logic.

Run: python stress_test.py

Bar frequency: 5-minute (inherits backtest.py default)
max_hold_bars: 12 bars = 60 min at 5-min resolution
"""
import pandas as pd
import numpy as np
from itertools import product
from backtest import load_day, resample_to_bars, backtest_mean_reversion

INSTRUMENT = "XAUUSD"
ALL_DATES = [
    "2026_05_19", "2026_05_20", "2026_05_21", "2026_05_22",
    "2026_05_25", "2026_05_26", "2026_05_27", "2026_05_28",
    "2026_06_01", "2026_06_02", "2026_06_03", "2026_06_04",
]

# XAUUSD typical mid price — used for std floor calibration, matches main.py
XAUUSD_MID = 3300.0


def _safe_concat(frames: list) -> pd.DataFrame:
    """Concat a list of DataFrames, silently dropping empty ones.
    Returns an empty DataFrame with the expected columns if nothing survives."""
    non_empty = [f for f in frames if f is not None and len(f) > 0]
    if not non_empty:
        return pd.DataFrame(columns=['ts', 'side', 'entry', 'exit',
                                     'pnl_bps', 'bars_held', 'exit_reason'])
    return pd.concat(non_empty, ignore_index=True)


def _sharpe(s: pd.Series) -> float:
    std = s.std()
    return float(s.mean() / std) if std > 0 else 0.0


def _profit_factor(s: pd.Series) -> float:
    wins   = s[s > 0].sum()
    losses = abs(s[s < 0].sum())
    return wins / losses if losses > 0 else float('inf')


# ── Step 1: Walk-Forward Validation ──────────────────────────────────────────

def walk_forward(bars_by_date: dict, train_days: int = 6, test_days: int = 2):
    """
    Rolling walk-forward: optimise on `train_days`, validate on `test_days`.
    Default windows (6 train / 2 test) produce 5 folds on 12 available dates.
    Returns a DataFrame of out-of-sample results per fold.
    """
    dates   = list(bars_by_date.keys())
    n_dates = len(dates)
    results = []

    n_folds = n_dates - train_days - test_days + 1
    if n_folds <= 0:
        print(f"  ⚠️  Not enough dates ({n_dates}) for train={train_days} + test={test_days}.")
        return pd.DataFrame()

    for start in range(n_folds):
        train_dates = dates[start : start + train_days]
        test_dates  = dates[start + train_days : start + train_days + test_days]

        train_bars = [bars_by_date[d] for d in train_dates]
        test_bars  = [bars_by_date[d] for d in test_dates]

        # ── Grid search on train set ──
        best_sharpe = -np.inf
        best_params: dict = {}

        for entry_z, lookback, sl in product([1.8, 2.0, 2.5], [10, 15, 20], [6.0, 8.0, 12.0]):
            raw = [
                backtest_mean_reversion(
                    b, entry_z=entry_z, lookback=lookback,
                    stop_loss_bps=sl, imbalance_threshold=-1.0,
                    regime_bias="BOTH", max_hold_bars=12,
                    mid_price_for_std_floor=XAUUSD_MID,
                )
                for b in train_bars
            ]
            combined = _safe_concat(raw)
            if len(combined) < 5:
                continue
            sh = _sharpe(combined['pnl_bps'])
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = dict(entry_z=entry_z, lookback=lookback, stop_loss_bps=sl)

        # ── Out-of-sample evaluation ──
        if not best_params:
            print(f"  Fold {start}: no valid train params found — skipping.")
            continue

        raw_oos = [
            backtest_mean_reversion(
                b, imbalance_threshold=-1.0, regime_bias="BOTH",
                max_hold_bars=12, mid_price_for_std_floor=XAUUSD_MID,
                **best_params,
            )
            for b in test_bars
        ]
        oos = _safe_concat(raw_oos)
        oos_sharpe = _sharpe(oos['pnl_bps']) if len(oos) >= 2 else 0.0
        oos_pnl    = float(oos['pnl_bps'].sum()) if len(oos) > 0 else 0.0
        oos_wr     = float((oos['pnl_bps'] > 0).mean()) if len(oos) > 0 else 0.0

        results.append({
            'fold':         start,
            'train_dates':  f"{train_dates[0]}..{train_dates[-1]}",
            'test_dates':   f"{test_dates[0]}..{test_dates[-1]}",
            'best_params':  str(best_params),
            'train_sharpe': best_sharpe,
            'oos_sharpe':   oos_sharpe,
            'oos_trades':   len(oos),
            'oos_pnl':      oos_pnl,
            'oos_win_rate': oos_wr,
        })

        print(f"  Fold {start} ({test_dates[0]}..{test_dates[-1]}): "
              f"Train={best_sharpe:.3f} → OOS={oos_sharpe:.3f}  "
              f"({len(oos)} trades, {oos_pnl:.1f} bps)  params={best_params}")

    return pd.DataFrame(results)


# ── Step 2: Monte Carlo Confidence Intervals ──────────────────────────────────

def monte_carlo_simulation(trades_df: pd.DataFrame, n_sims: int = 10_000):
    """
    Randomly reshuffle trade PnL to generate confidence intervals on terminal wealth.
    If actual terminal PnL < 50th percentile of shuffled runs, the strategy has no edge.
    """
    if len(trades_df) == 0:
        print("  No trades to simulate.")
        return

    pnls: np.ndarray = trades_df['pnl_bps'].to_numpy(dtype=float)
    actual_total     = float(pnls.sum())

    sim_totals       = np.empty(n_sims)
    sim_max_dd       = np.empty(n_sims)

    for k in range(n_sims):
        shuffled        = np.random.permutation(pnls)
        sim_totals[k]   = shuffled.sum()
        cumulative      = np.cumsum(shuffled)
        running_max     = np.maximum.accumulate(cumulative)
        sim_max_dd[k]   = (running_max - cumulative).max()

    print(f"\n  {'='*50}")
    print(f"  MONTE CARLO ANALYSIS ({n_sims:,} simulations)")
    print(f"  {'='*50}")
    print(f"  Actual total PnL:   {actual_total:+.2f} bps")
    print(f"  MC median PnL:      {np.median(sim_totals):+.2f} bps")
    print(f"  MC 5th percentile:  {np.percentile(sim_totals, 5):+.2f} bps")
    print(f"  MC 95th percentile: {np.percentile(sim_totals, 95):+.2f} bps")
    print(f"  MC 99th pctl DD:    {np.percentile(sim_max_dd, 99):.2f} bps")
    print(f"  P(actual > median): {(sim_totals < actual_total).mean():.1%}")

    actual_cum = np.cumsum(pnls)  # pnls already np.ndarray[float] from above
    actual_rm  = np.maximum.accumulate(actual_cum)
    actual_dd  = float((actual_rm - actual_cum).max())
    dd_pctl    = float((sim_max_dd < actual_dd).mean())
    print(f"  Actual max DD:      {actual_dd:.2f} bps  "
          f"(worse than {dd_pctl:.0%} of shuffled runs)")

    edge_flag = "✅ Edge confirmed" if (sim_totals < actual_total).mean() > 0.75 else "⚠️  Weak edge — review"
    print(f"  Verdict:            {edge_flag}")


# ── Step 3: Parameter Sensitivity ────────────────────────────────────────────

def parameter_sensitivity(all_bars: list):
    """Run P1–P5 sensitivity sweeps and flag cliff edges (Sharpe jumps > 0.15)."""

    def _run(bars_list, max_hold_bars=12, **kwargs) -> pd.DataFrame:
        """Run backtest across all days with given kwargs, return combined trades."""
        raw = [
            backtest_mean_reversion(
                b, imbalance_threshold=-1.0, regime_bias="BOTH",
                max_hold_bars=max_hold_bars, mid_price_for_std_floor=XAUUSD_MID,
                **kwargs,
            )
            for b in bars_list
        ]
        return _safe_concat(raw)

    print(f"\n  {'='*70}")
    print(f"  PARAMETER SENSITIVITY ANALYSIS  (5-min bars, XAUUSD, BOTH regime)")
    print(f"  {'='*70}")

    # ── P1: Lookback ──
    print(f"\n  --- P1: Lookback (entry_z=2.0, SL=6.0) ---")
    print(f"  {'LB':>4}  {'Trades':>6}  {'PnL':>9}  {'Sharpe':>7}  {'WR':>6}  {'SLhits':>6}  {'Verdict'}")
    prev_sh = None
    for lb in [5, 8, 10, 12, 15, 20, 30]:
        t = _run(all_bars, entry_z=2.0, lookback=lb, stop_loss_bps=6.0)
        if len(t) == 0:
            print(f"  {lb:>4}  {'—':>6}")
            continue
        sh    = _sharpe(t['pnl_bps'])
        sl_h  = (t['exit_reason'] == 'SL').sum()
        wr    = (t['pnl_bps'] > 0).mean()
        v     = ""
        if prev_sh is not None:
            delta = sh - prev_sh
            if abs(delta) > 0.15:
                v = "⚠️  CLIFF" if delta < 0 else "📈 JUMP"
        prev_sh = sh
        print(f"  {lb:>4}  {len(t):>6}  {t['pnl_bps'].sum():>+9.2f}  "
              f"{sh:>7.4f}  {wr:>6.1%}  {sl_h:>6}  {v}")

    # ── P2: Entry Z ──
    print(f"\n  --- P2: Entry Z (LB=20, SL=6.0) ---")
    print(f"  {'Z':>5}  {'Trades':>6}  {'PnL':>9}  {'Sharpe':>7}  {'PnL/Trade':>10}")
    for ez in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5]:
        t = _run(all_bars, entry_z=ez, lookback=20, stop_loss_bps=6.0)
        if len(t) == 0:
            print(f"  {ez:>5.1f}  {'—':>6}")
            continue
        sh = _sharpe(t['pnl_bps'])
        print(f"  {ez:>5.1f}  {len(t):>6}  {t['pnl_bps'].sum():>+9.2f}  "
              f"{sh:>7.4f}  {t['pnl_bps'].mean():>+10.4f}")

    # ── P3: Exit Z ──
    print(f"\n  --- P3: Exit Z (LB=20, entry_z=2.0, SL=6.0) ---")
    print(f"  {'ExitZ':>6}  {'Trades':>6}  {'PnL':>9}  {'Sharpe':>7}  {'AvgBars':>8}")
    for xz in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
        t = _run(all_bars, entry_z=2.0, exit_z=xz, lookback=20, stop_loss_bps=6.0)
        if len(t) == 0:
            print(f"  {xz:>6.1f}  {'—':>6}")
            continue
        sh = _sharpe(t['pnl_bps'])
        print(f"  {xz:>6.1f}  {len(t):>6}  {t['pnl_bps'].sum():>+9.2f}  "
              f"{sh:>7.4f}  {t['bars_held'].mean():>8.1f}")

    # ── P4: Stop Loss ──
    print(f"\n  --- P4: Stop Loss (LB=20, entry_z=2.0) ---")
    print(f"  {'SL':>5}  {'Trades':>6}  {'PnL':>9}  {'Sharpe':>7}  {'SLhits':>6}  {'PF':>6}")
    for sl in [3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        t = _run(all_bars, entry_z=2.0, lookback=20, stop_loss_bps=sl)
        if len(t) == 0:
            print(f"  {sl:>5.1f}  {'—':>6}")
            continue
        sh   = _sharpe(t['pnl_bps'])
        pf   = _profit_factor(t['pnl_bps'])
        sl_h = (t['exit_reason'] == 'SL').sum()
        print(f"  {sl:>5.1f}  {len(t):>6}  {t['pnl_bps'].sum():>+9.2f}  "
              f"{sh:>7.4f}  {sl_h:>6}  {pf:>6.3f}")

    # ── P5: Regime filter ──
    print(f"\n  --- P5: Regime Filter (LB=20, entry_z=2.0, SL=6.0) ---")
    print(f"  {'Bias':>8}  {'Trades':>6}  {'PnL':>9}  {'Sharpe':>7}  {'PF':>6}  {'WR':>6}")
    for bias in ["BEARISH", "BULLISH", "BOTH"]:
        raw = [
            backtest_mean_reversion(
                b, entry_z=2.0, lookback=20, stop_loss_bps=6.0,
                imbalance_threshold=-1.0, regime_bias=bias,
                max_hold_bars=12, mid_price_for_std_floor=XAUUSD_MID,
            )
            for b in all_bars
        ]
        t = _safe_concat(raw)
        if len(t) == 0:
            print(f"  {bias:>8}  {'—':>6}")
            continue
        sh = _sharpe(t['pnl_bps'])
        pf = _profit_factor(t['pnl_bps'])
        wr = (t['pnl_bps'] > 0).mean()
        print(f"  {bias:>8}  {len(t):>6}  {t['pnl_bps'].sum():>+9.2f}  "
              f"{sh:>7.4f}  {pf:>6.3f}  {wr:>6.1%}")

    # ── P6: Max hold bars ──
    print(f"\n  --- P6: Max Hold Bars (LB=20, entry_z=2.0, SL=6.0) ---")
    print(f"  {'Hold':>5}  {'(mins)':>7}  {'Trades':>6}  {'PnL':>9}  {'Sharpe':>7}  {'TimeExits':>9}")
    for hold in [3, 6, 8, 12, 18, 24]:
        t = _run(all_bars, max_hold_bars=hold, entry_z=2.0, lookback=20, stop_loss_bps=6.0)
        if len(t) == 0:
            print(f"  {hold:>5}  {'—':>7}")
            continue
        sh   = _sharpe(t['pnl_bps'])
        te   = (t['exit_reason'] == 'Time').sum()
        mins = hold * 5
        print(f"  {hold:>5}  {mins:>6}m  {len(t):>6}  {t['pnl_bps'].sum():>+9.2f}  "
              f"{sh:>7.4f}  {te:>9}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  STRATEGY STRESS TEST SUITE")
    print("  XAUUSD | 5-min bars | 12-date sample (May–Jun 2026)")
    print("=" * 70)

    print(f"\n📂 Loading {len(ALL_DATES)} days of tick data...")
    bars_by_date: dict = {}
    all_bars: list     = []

    for d in ALL_DATES:
        try:
            bars = resample_to_bars(load_day(INSTRUMENT, d))
            if len(bars) == 0:
                print(f"  ⚠️  {d} — 0 bars after resampling, skipping.")
                continue
            bars_by_date[d] = bars
            all_bars.append(bars)
            print(f"  ✅ {d}  ({len(bars)} bars)")
        except Exception as e:
            print(f"  ⚠️  {d} SKIP — {e}")

    if not all_bars:
        print("🚨 No data loaded. Aborting.")
        exit(1)

    print(f"\n  {len(all_bars)} days loaded successfully.")

    # 1. Walk-forward (6 train / 2 test → 5 folds on 12 dates)
    print(f"\n\n{'─'*70}")
    print(f"▶  WALK-FORWARD VALIDATION  (train=6 days, test=2 days, 5 folds)")
    print(f"{'─'*70}\n")
    wf = walk_forward(bars_by_date, train_days=6, test_days=2)
    if len(wf) > 0:
        avg_train = wf['train_sharpe'].mean()
        avg_oos   = wf['oos_sharpe'].mean()
        decay     = avg_train - avg_oos
        overfit   = "⚠️  OVERFIT" if avg_train > 2 * avg_oos else "✅ Acceptable"
        print(f"\n  Summary across {len(wf)} folds:")
        print(f"  Avg train Sharpe:  {avg_train:.4f}")
        print(f"  Avg OOS Sharpe:    {avg_oos:.4f}")
        print(f"  Avg OOS PnL:       {wf['oos_pnl'].mean():.2f} bps")
        print(f"  Sharpe decay:      {decay:.4f}  ({overfit})")
        print(f"  OOS win rate:      {wf['oos_win_rate'].mean():.1%}")

    # 2. Monte Carlo
    print(f"\n\n{'─'*70}")
    print(f"▶  MONTE CARLO ANALYSIS  (10,000 simulations)")
    print(f"{'─'*70}")
    mc_raw = [
        backtest_mean_reversion(
            b, entry_z=2.0, lookback=20, stop_loss_bps=6.0,
            imbalance_threshold=-1.0, regime_bias="BOTH",
            max_hold_bars=12, mid_price_for_std_floor=XAUUSD_MID,
        )
        for b in all_bars
    ]
    full_df = _safe_concat(mc_raw)
    monte_carlo_simulation(full_df)

    # 3. Parameter sensitivity
    print(f"\n\n{'─'*70}")
    print(f"▶  PARAMETER SENSITIVITY  (P1–P6)")
    print(f"{'─'*70}")
    parameter_sensitivity(all_bars)

    print(f"\n\n{'='*70}")
    print(f"  ✅ STRESS TEST COMPLETE")
    print(f"{'='*70}")