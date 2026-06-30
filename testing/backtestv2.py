import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

DATA_DIR = Path("pricer-output-2026-05-11_2026-06-10/")
con = duckdb.connect()

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_day(instrument: str, date: str) -> pd.DataFrame:
    path = DATA_DIR / f"{instrument}_{date}.parquet"
    df = con.execute("""
        SELECT time, bid, ask, (bid + ask) / 2 AS mid, ask - bid AS spread
        FROM read_parquet($path) ORDER BY time
    """, {"path": str(path)}).df()
    df['time'] = pd.to_datetime(df['time'])
    return df

def resample_to_bars(df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    df = df.set_index('time')
    bars = df['mid'].resample(freq).ohlc()
    bars['spread_mean'] = df['spread'].resample(freq).mean()
    return bars.dropna()

# ── Strategy ──────────────────────────────────────────────────────────────────

def backtest_statistical_scalper(
    bars: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 2.2,
    max_hold_bars: int = 10
) -> pd.DataFrame:
    """
    Z-Score Mean Reversion Scalper.
    Fixed: bars_held off-by-one, same-bar re-entry, volatility floor as param.
    """
    bars = bars.copy()
    bars['mean'] = bars['close'].rolling(lookback).mean()
    bars['std']  = bars['close'].rolling(lookback).std()

    # Volatility floor: use 10% of rolling mean std as floor (adaptive, not hardcoded)
    vol_floor = bars['std'].rolling(lookback).mean().fillna(0.15) * 0.1
    vol_floor = vol_floor.clip(lower=0.05)
    bars['z_score'] = (bars['close'] - bars['mean']) / np.maximum(bars['std'], vol_floor)

    opens      = bars['open'].to_numpy()
    closes     = bars['close'].to_numpy()
    means      = bars['mean'].to_numpy()
    zs         = bars['z_score'].to_numpy()
    spreads    = bars['spread_mean'].to_numpy()
    timestamps = bars.index.to_list()

    position   = 0
    entry_price = 0.0
    entry_idx  = 0
    just_exited = False   # FIX: prevent same-bar re-entry
    trades     = []

    for i in range(lookback, len(bars) - 1):
        # FIX: bars_held counts from the bar AFTER entry (entry_idx = i+1, so first check is at i=entry_idx)
        bars_held = i - entry_idx

        just_exited = False

        # ─── 1. EXIT LOGIC ───
        if position == 1:
            if closes[i] >= means[i] or bars_held >= max_hold_bars:
                exit_price = opens[i + 1] - (spreads[i + 1] / 2)
                pnl = exit_price - entry_price
                trades.append({
                    'ts': timestamps[i + 1], 'side': 'long',
                    'entry': entry_price, 'exit': exit_price,
                    'pnl_pts': pnl, 'bars_held': bars_held,
                    'reason': 'Mean Reverted' if closes[i] >= means[i] else 'Time Expired'
                })
                position = 0
                just_exited = True

        elif position == -1:
            if closes[i] <= means[i] or bars_held >= max_hold_bars:
                exit_price = opens[i + 1] + (spreads[i + 1] / 2)
                pnl = entry_price - exit_price
                trades.append({
                    'ts': timestamps[i + 1], 'side': 'short',
                    'entry': entry_price, 'exit': exit_price,
                    'pnl_pts': pnl, 'bars_held': bars_held,
                    'reason': 'Mean Reverted' if closes[i] <= means[i] else 'Time Expired'
                })
                position = 0
                just_exited = True

        # ─── 2. ENTRY LOGIC ───
        if position == 0 and not just_exited:   # FIX: no same-bar re-entry
            if zs[i] <= -entry_z:
                position  = 1
                entry_idx = i + 1
                entry_price = opens[i + 1] + (spreads[i + 1] / 2)
            elif zs[i] >= entry_z:
                position  = -1
                entry_idx = i + 1
                entry_price = opens[i + 1] - (spreads[i + 1] / 2)

    return pd.DataFrame(trades)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {k: 0 for k in ['n_trades','win_rate','total_pts','profit_factor',
                                'sharpe_15m','avg_hold','pct_time_expired']}
    wins   = trades[trades['pnl_pts'] > 0]['pnl_pts'].sum()
    losses = trades[trades['pnl_pts'] < 0]['pnl_pts'].sum()
    pf     = wins / abs(losses) if losses != 0 else float('inf')
    trades = trades.copy()
    trades['ts'] = pd.to_datetime(trades['ts'])
    window_pnl  = trades.set_index('ts')['pnl_pts'].resample('15min').sum().fillna(0)
    sharpe      = window_pnl.mean() / window_pnl.std() if window_pnl.std() > 0 else 0
    pct_expired = (trades['reason'] == 'Time Expired').mean()
    return {
        'n_trades':        len(trades),
        'win_rate':        (trades['pnl_pts'] > 0).mean(),
        'total_pts':       trades['pnl_pts'].sum(),
        'profit_factor':   pf,
        'sharpe_15m':      sharpe,
        'avg_hold':        trades['bars_held'].mean(),
        'pct_time_expired': pct_expired,
    }

# ── Per-Day Report ────────────────────────────────────────────────────────────

def run_perday(instrument, dates, lookback=20, entry_z=2.2, max_hold_bars=10):
    rows = []
    for date in dates:
        try:
            bars = resample_to_bars(load_day(instrument, date))
            res  = backtest_statistical_scalper(bars, lookback, entry_z, max_hold_bars)
            m    = compute_metrics(res)
            m['date'] = date
            rows.append(m)
        except Exception as e:
            print(f"  [!] {date} failed: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index('date')
    return df

def print_perday(df, label="Z-SCORE REVERSION SCALPER"):
    if df.empty:
        print("No data."); return
    totals = compute_metrics(pd.concat(_cache, ignore_index=True)) if _cache else {}
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(df[['n_trades','win_rate','total_pts','profit_factor','sharpe_15m','pct_time_expired']].to_string(
        float_format=lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:+.1f}"
    ))

    # Aggregate row
    agg = df.agg({
        'n_trades': 'sum', 'win_rate': 'mean', 'total_pts': 'sum',
        'profit_factor': 'mean', 'sharpe_15m': 'mean', 'pct_time_expired': 'mean'
    })
    print(f"{'─'*65}")
    print(f"  TOTAL/AVG  trades={int(agg.n_trades)}  win%={agg.win_rate:.1%}  "
          f"pts={agg.total_pts:+.2f}  PF={agg.profit_factor:.3f}  "
          f"15m-Sharpe={agg.sharpe_15m:.4f}  %expired={agg.pct_time_expired:.1%}")

# ── Parameter Sweep ───────────────────────────────────────────────────────────

def param_sweep(instrument, dates, 
                entry_z_values   = [1.6, 1.8, 2.0, 2.2, 2.5, 2.8],
                max_hold_values  = [5, 8, 10, 15, 20],
                lookback         = 20):
    print(f"\n{'='*65}")
    print(f"  PARAMETER SWEEP  (lookback={lookback})")
    print(f"{'='*65}")
    print(f"  {'entry_z':>8}  {'max_hold':>9}  {'trades':>7}  {'win%':>7}  "
          f"{'total_pts':>10}  {'PF':>7}  {'sharpe':>8}  {'%expired':>9}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*9}")

    results = []
    all_bars = {}
    for date in dates:
        try:
            all_bars[date] = resample_to_bars(load_day(instrument, date))
        except Exception as e:
            print(f"  [!] Could not load {date}: {e}")

    if not all_bars:
        print("  No data loaded."); return pd.DataFrame()

    for ez, mh in product(entry_z_values, max_hold_values):
        all_trades = []
        for date, bars in all_bars.items():
            res = backtest_statistical_scalper(bars, lookback=lookback, entry_z=ez, max_hold_bars=mh)
            if not res.empty:
                all_trades.append(res)
        combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        m = compute_metrics(combined)
        m.update({'entry_z': ez, 'max_hold': mh})
        results.append(m)
        marker = " ◀ default" if ez == 2.2 and mh == 10 else ""
        print(f"  {ez:>8.1f}  {mh:>9}  {m['n_trades']:>7}  {m['win_rate']:>7.1%}  "
              f"{m['total_pts']:>+10.2f}  {m['profit_factor']:>7.3f}  "
              f"{m['sharpe_15m']:>8.4f}  {m['pct_time_expired']:>9.1%}{marker}")

    sweep_df = pd.DataFrame(results).sort_values('sharpe_15m', ascending=False)
    print(f"\n{'─'*65}")
    print("  TOP 5 BY 15-MIN SHARPE:")
    for _, r in sweep_df.head(5).iterrows():
        print(f"    entry_z={r.entry_z:.1f}  max_hold={int(r.max_hold)}  "
              f"pts={r.total_pts:+.2f}  win%={r.win_rate:.1%}  "
              f"sharpe={r.sharpe_15m:.4f}  PF={r.profit_factor:.3f}")
    print(f"\n  TOP 5 BY TOTAL POINTS:")
    for _, r in sweep_df.sort_values('total_pts', ascending=False).head(5).iterrows():
        print(f"    entry_z={r.entry_z:.1f}  max_hold={int(r.max_hold)}  "
              f"pts={r.total_pts:+.2f}  win%={r.win_rate:.1%}  "
              f"sharpe={r.sharpe_15m:.4f}  PF={r.profit_factor:.3f}")
    return sweep_df

# ── Main ──────────────────────────────────────────────────────────────────────

INSTRUMENT = "XAGUSD"
DATES = [
    "2026_05_19", "2026_05_20", "2026_05_21", "2026_05_22", "2026_05_25",
    "2026_05_26", "2026_05_27", "2026_05_28", "2026_06_01", "2026_06_02",
    "2026_06_03", "2026_06_04",
]

print("Running per-day report (default params: entry_z=2.2, max_hold=10)...")
_cache = []  # store trades for aggregate
perday_rows = []
for date in DATES:
    try:
        bars = resample_to_bars(load_day(INSTRUMENT, date))
        res  = backtest_statistical_scalper(bars, lookback=20, entry_z=2.2, max_hold_bars=10)
        if not res.empty:
            _cache.append(res)
        m = compute_metrics(res)
        m['date'] = date
        perday_rows.append(m)
    except Exception as e:
        print(f"  [!] {date}: {e}")

if perday_rows:
    df = pd.DataFrame(perday_rows).set_index('date')
    print(f"\n{'='*65}")
    print(f"  Z-SCORE REVERSION SCALPER — PER DAY (entry_z=2.2, max_hold=10)")
    print(f"{'='*65}")
    print(df[['n_trades','win_rate','total_pts','profit_factor','sharpe_15m','pct_time_expired']].to_string())
    if _cache:
        all_trades = pd.concat(_cache, ignore_index=True)
        agg = compute_metrics(all_trades)
        print(f"{'─'*65}")
        print(f"  AGGREGATE  trades={agg['n_trades']}  win%={agg['win_rate']:.1%}  "
              f"pts={agg['total_pts']:+.2f}  PF={agg['profit_factor']:.3f}  "
              f"15m-Sharpe={agg['sharpe_15m']:.4f}  %expired={agg['pct_time_expired']:.1%}")

print("\n\nRunning parameter sweep...")
sweep = param_sweep(INSTRUMENT, DATES)