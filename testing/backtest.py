import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("pricer-output-2026-05-11_2026-06-10/")
con = duckdb.connect()

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

# ── High-Sharpe Statistical Mean Reversion Strategy ───────────────────────────

def backtest_statistical_scalper(
    bars: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 2.2,     # Trigger trade at 2.2 Standard Deviations out
    max_hold_bars: int = 10   # Stale trade protection (10 minutes max)
) -> pd.DataFrame:
    """
    High-Sharpe Scalper focused on capturing rapid mean reversions.
    - No static SL/TP brackets.
    - Enters via Market order on extreme statistical overextensions.
    - Exits safely the moment price snaps back to its rolling average.
    """
    bars = bars.copy()
    
    # Calculate rolling statistical bounds
    bars['mean'] = bars['close'].rolling(lookback).mean()
    bars['std'] = bars['close'].rolling(lookback).std()
    
    # Prevent divide-by-zero on flat bars using a volatility floor
    bars['z_score'] = (bars['close'] - bars['mean']) / np.maximum(bars['std'], 0.15)
    
    opens = bars['open'].to_numpy()
    closes = bars['close'].to_numpy()
    means = bars['mean'].to_numpy()
    zs = bars['z_score'].to_numpy()
    spreads = bars['spread_mean'].to_numpy()
    timestamps = bars.index.to_list()
    
    position = 0 # 1 = Long, -1 = Short
    entry_price = 0.0
    entry_idx = 0
    trades = []
    
    for i in range(lookback, len(bars) - 1):
        bars_held = i - entry_idx
        
        # ─── 1. EVALUATE REVERSION EXITS ───
        if position == 1:
            # Exit long as soon as price recovers back to or above the rolling mean
            if closes[i] >= means[i] or bars_held >= max_hold_bars:
                exit_price = opens[i + 1] - (spreads[i + 1] / 2)
                pnl = exit_price - entry_price
                trades.append({
                    'ts': timestamps[i], 'side': 'long', 'entry': entry_price, 'exit': exit_price,
                    'pnl_usd': pnl, 'bars_held': bars_held + 1,
                    'reason': 'Mean Reverted' if closes[i] >= means[i] else 'Time Expired'
                })
                position = 0
                
        elif position == -1:
            # Exit short as soon as price drops back to or below the rolling mean
            if closes[i] <= means[i] or bars_held >= max_hold_bars:
                exit_price = opens[i + 1] + (spreads[i + 1] / 2)
                pnl = entry_price - exit_price
                trades.append({
                    'ts': timestamps[i], 'side': 'short', 'entry': entry_price, 'exit': exit_price,
                    'pnl_usd': pnl, 'bars_held': bars_held + 1,
                    'reason': 'Mean Reverted' if closes[i] <= means[i] else 'Time Expired'
                })
                position = 0

        # ─── 2. EVALUATE OVEREXTENSION ENTRIES ───
        if position == 0:
            # Price is crushed outward below bands -> Buy the rubber-band snapback
            if zs[i] <= -entry_z:
                position = 1
                entry_idx = i + 1
                entry_price = opens[i + 1] + (spreads[i + 1] / 2)
                
            # Price is pumped outward above bands -> Short the rubber-band snapback
            elif zs[i] >= entry_z:
                position = -1
                entry_idx = i + 1
                entry_price = opens[i + 1] - (spreads[i + 1] / 2)

    return pd.DataFrame(trades)

def print_results(results: pd.DataFrame, label: str):
    if len(results) == 0:
        print(f"\n[{label}] No trades executed."); return

    wins = results[results['pnl_usd'] > 0]['pnl_usd'].sum()
    losses = results[results['pnl_usd'] < 0]['pnl_usd'].sum()
    pf = wins / abs(losses) if losses != 0 else float('inf')
    
    # Calculate 15-minute Sharpe ratio
    results['ts'] = pd.to_datetime(results['ts'])
    ts_df = results.set_index('ts')
    window_pnl = ts_df['pnl_usd'].resample('15min').sum().fillna(0)
    sharpe_15m = window_pnl.mean() / window_pnl.std() if window_pnl.std() > 0 else 0

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"Total trades executed:  {len(results)}")
    print(f"Win rate:               {(results['pnl_usd'] > 0).mean():.1%}")
    print(f"Total Points PnL ($):   {results['pnl_usd'].sum():+.2f}")
    print(f"Profit factor:          {pf:.3f}")
    print(f"15-min Sharpe Ratio:    {sharpe_15m:.4f}")
    print(f"Avg trade duration:     {results['bars_held'].mean():.1f} bar(s)")
    print("\nExit Breakdown:")
    print(results['reason'].value_counts().to_string())

# ── Run Evaluation ────────────────────────────────────────────────────────────
INSTRUMENT = "XAUUSD"
DATES = ["2026_05_19", "2026_05_20", "2026_05_21", "2026_05_22", "2026_05_25"]

all_trades = []
for date in DATES:
    try:
        bars = resample_to_bars(load_day(INSTRUMENT, date), freq='1min')
        res = backtest_statistical_scalper(bars, lookback=20, entry_z=2.2)
        all_trades.append(res)
    except Exception as e: pass

if all_trades:
    print_results(pd.concat(all_trades, ignore_index=True), "STATISTICAL Z-SCORE REVERSION SCALPER")