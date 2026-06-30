"""
dca_backtest.py — Trend Pullback Scalper (LIMIT ORDER & EXPLICIT TP)
===================================================================
Single-bullet execution (No DCA). 
Places a LIMIT BUY at the 20-MA when armed (z > 2.0). 
Order remains valid for 10 bars. Fills intra-bar if Low <= Limit Price.
Asymmetric 1:2 Risk/Reward ($1,000 risk vs $2,000 target).
Explicit Take Profit and Stop Loss price levels calculated on fill.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── Trend Pullback Config ─────────────────────────────────────────────────────

LOOKBACK     = 20
ENTRY_Z      = 2.0           # Triggers the "ARMED" state
EXIT_Z       = 2.0           # Dynamic profit target extended to upper 2.0 SD band
ARMED_WINDOW = 10            # Bars the limit order remains active
BAR_FREQ_MIN = 5
ACCOUNT_SIZE = 1_000_000.0

DCA_CONFIG = {
    "XAUUSD": {
        "volume_base":   0.5,        # SURVIVAL SIZING: 0.5 Lots ($50/pt)
        "point_value":   100.0,
        "max_tranches":  1,          # SINGLE BULLET: No DCA scaling
        "profit_target": 2000.0,     # ASYMMETRIC TARGET: 1:2 Risk/Reward
        "emergency_sl":  -1000.0,    # SYMMETRIC HARD STOP: Exact $1k fill
    },
    "XAGUSD": {
        "volume_base":   0.3,        
        "point_value":   50.0,
        "max_tranches":  1,
        "profit_target": 2000.0,      
        "emergency_sl":  -1000.0,
    },
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def _parquet_to_mid_series(path: str, chunk_size: int) -> pd.Series:
    import pyarrow.parquet as pq
    pf    = pq.ParquetFile(path)
    parts = []
    for batch in pf.iter_batches(batch_size=chunk_size, columns=["time", "bid", "ask"]):
        d     = batch.to_pydict()
        times = pd.to_datetime(d["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        mids  = (np.array(d["bid"], dtype=float) + np.array(d["ask"], dtype=float)) / 2.0
        parts.append(pd.Series(mids, index=times))
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    return s[s.index.notna()]


def load_folder_to_ohlc(folder: str, symbol: str, chunk_size: int) -> pd.DataFrame:
    pattern = os.path.join(folder, f"*{symbol}*.parquet")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .parquet files matching {symbol} found in: {folder}")
    all_parts = []
    for i, fpath in enumerate(files, 1):
        s = _parquet_to_mid_series(fpath, chunk_size)
        all_parts.append(s)
    combined = pd.concat(all_parts).sort_index()
    ohlc          = combined.resample(f"{BAR_FREQ_MIN}min").ohlc().dropna()
    ohlc.columns  = ["open", "high", "low", "close"]
    ohlc["volume"] = combined.resample(f"{BAR_FREQ_MIN}min").count()
    return ohlc


def load_single_parquet_to_ohlc(path: str, chunk_size: int) -> pd.DataFrame:
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = f"/mnt/user-data/uploads/{path}"
        if os.path.exists(candidate):
            path = candidate
    s             = _parquet_to_mid_series(path, chunk_size)
    ohlc          = s.resample(f"{BAR_FREQ_MIN}min").ohlc().dropna()
    ohlc.columns  = ["open", "high", "low", "close"]
    ohlc["volume"] = s.resample(f"{BAR_FREQ_MIN}min").count()
    return ohlc


def load_csv_to_ohlc(path: str) -> pd.DataFrame:
    df         = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.lower() for c in df.columns]
    return df


def generate_synthetic_ohlc(symbol: str, n_bars: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    base   = 3300.0 if symbol == "XAUUSD" else 33.0
    sigma  = base * 0.0015
    closes = [base]
    for _ in range(n_bars - 1):
        closes.append(closes[-1] + rng.normal(0, sigma))
    ts     = pd.date_range("2025-01-01", periods=n_bars, freq=f"{BAR_FREQ_MIN}min")
    arr    = np.array(closes)
    noise  = np.abs(rng.normal(0, sigma * 0.6, n_bars))
    return pd.DataFrame({
        "open": arr, "high": arr + noise, "low": arr - noise,
        "close": arr, "volume": 100,
    }, index=ts)


@dataclass
class Tranche:
    entry_price: float
    lots:        float

@dataclass
class Sequence:
    symbol:       str
    tranches:     list[Tranche] = field(default_factory=list)
    last_px:      float         = 0.0
    open_ts:      Optional[datetime] = None
    close_ts:     Optional[datetime] = None
    close_price:  float         = 0.0
    realized_pnl: float         = 0.0
    closed:       bool          = False
    
    # Explicit Limit Order Target Tracking
    tp_price:         float     = 0.0
    sl_price:         float     = 0.0
    
    breakeven_locked: bool      = False
    bars_held:        int       = 0

    @property
    def total_lots(self) -> float:
        return sum(t.lots for t in self.tranches)

    @property
    def avg_entry(self) -> float:
        if not self.tranches:
            return 0.0
        return sum(t.entry_price * t.lots for t in self.tranches) / self.total_lots

    def unrealized_pnl(self, price: float) -> float:
        cfg = DCA_CONFIG[self.symbol]
        return self.total_lots * cfg["point_value"] * (price - self.avg_entry)


# ── Core Engine (Intra-Bar Exact Fills + Limit Orders) ────────────────────────

def run_backtest(symbol: str, df: pd.DataFrame, verbose: bool = False) -> dict:
    cfg        = DCA_CONFIG[symbol]
    
    closes     = df["close"].to_numpy(dtype=float)
    highs      = df["high"].to_numpy(dtype=float)
    lows       = df["low"].to_numpy(dtype=float)
    volumes    = df["volume"].to_numpy(dtype=float)
    timestamps = df.index

    vol_sma = df["volume"].rolling(window=LOOKBACK).mean().to_numpy()

    equity       = ACCOUNT_SIZE
    sequences:   list[Sequence] = []
    current:     Optional[Sequence] = None
    equity_curve = [equity]

    trigger_armed_bars  = -1
    pending_limit_price = None

    for i in range(LOOKBACK, len(closes)):
        c_price = closes[i]
        h_price = highs[i]
        l_price = lows[i]
        ts      = timestamps[i]
        vol     = volumes[i]
        v_sma   = vol_sma[i]

        window = closes[i - LOOKBACK: i]
        ma     = window.mean()
        std    = window.std(ddof=1)
        if std == 0:
            equity_curve.append(equity + (current.unrealized_pnl(c_price) if current else 0.0))
            continue
            
        z = (c_price - ma) / std

        # ── 1. Exit Tracking (Evaluating TP/SL Limits) ──
        if current is not None:
            current.bars_held += 1
            pnl_close = current.unrealized_pnl(c_price)
            
            # Activate Trailing Breakeven (Relaxed to +$1000 / 1R)
            be_trigger_price = current.avg_entry + (1000.0 / (current.total_lots * cfg["point_value"]))
            if h_price >= be_trigger_price and not current.breakeven_locked:
                current.breakeven_locked = True
                current.sl_price = current.avg_entry # Move limit SL to break-even

            close_reason = None
            exact_price  = 0.0
            
            # Condition 1: Stop Loss Limit Hit (Emergency or Breakeven)
            if l_price <= current.sl_price:
                close_reason = "STOP LOSS HIT"
                exact_price  = current.sl_price
                
            # Condition 2: Take Profit Limit Hit
            elif h_price >= current.tp_price:
                close_reason = "TAKE PROFIT HIT"
                exact_price  = current.tp_price
                
            # Condition 3: Band-to-Band Dynamic Target Hit Intra-Bar
            elif ((h_price - ma) / std) >= EXIT_Z:
                close_reason = "BAND TARGET HIT"
                exact_price  = ma + (EXIT_Z * std)
                if current.unrealized_pnl(exact_price) < pnl_close:
                    exact_price = c_price
                
            # Condition 4: Time-Based Hard Stop (48 bars)
            elif current.bars_held >= 48:
                close_reason = "TIME STOP (STALLED TREND)"
                exact_price  = c_price

            # Execute Sequence Close
            if close_reason:
                exact_pnl = current.unrealized_pnl(exact_price)
                current.close_price  = exact_price
                current.close_ts     = ts
                current.realized_pnl = exact_pnl
                current.closed       = True
                equity  += exact_pnl
                
                if verbose:
                    print(f"  [{close_reason}] {ts} | PnL: ${exact_pnl:,.2f} | Bars: {current.bars_held}")
                    
                current = None

        # ── 2. Limit Order Entry Logic ──
        if current is None and pending_limit_price is not None:
            if l_price <= pending_limit_price:
                # Limit order triggered and filled intra-bar
                fill_price = pending_limit_price
                
                current = Sequence(symbol=symbol, open_ts=ts, last_px=fill_price)
                current.tranches.append(Tranche(entry_price=fill_price, lots=cfg["volume_base"]))
                
                # Set static TP and SL targets relative to fill price
                points_per_lot = cfg["volume_base"] * cfg["point_value"]
                tp_points = cfg["profit_target"] / points_per_lot
                sl_points = abs(cfg["emergency_sl"]) / points_per_lot
                
                current.tp_price = fill_price + tp_points
                current.sl_price = fill_price - sl_points
                
                current.last_px = c_price
                sequences.append(current)
                
                # Clear pending order status
                pending_limit_price = None
                trigger_armed_bars  = -1

        # ── 3. Arming the Strategy / Placing the Limit Order ──
        if trigger_armed_bars >= 0:
            trigger_armed_bars += 1
            if trigger_armed_bars > ARMED_WINDOW:
                trigger_armed_bars  = -1  
                pending_limit_price = None  # Cancel limit order if window expires
                
        # If trend pushes strongly (z > 2.0), place a working limit order at the Mean
        if z > ENTRY_Z and vol > v_sma:
            trigger_armed_bars  = 0
            pending_limit_price = ma  

        mtm = equity + (current.unrealized_pnl(c_price) if current else 0.0)
        equity_curve.append(mtm)

    # Final wrap-up
    if current is not None and not current.closed:
        p   = closes[-1]
        pnl = current.unrealized_pnl(p)
        current.close_price  = p
        current.close_ts     = timestamps[-1]
        current.realized_pnl = pnl
        current.closed       = True
        equity += pnl

    closed  = [s for s in sequences if s.closed]
    profits = [s.realized_pnl for s in closed]
    wins    = [p for p in profits if p > 0]

    ec     = np.array(equity_curve)
    ret    = np.diff(ec) / ec[:-1]
    bars_y = 288 * 252
    sharpe = (np.mean(ret) / np.std(ret) * math.sqrt(bars_y)) if np.std(ret) > 0 else 0.0
    peak   = np.maximum.accumulate(ec)
    dd     = (peak - ec) / peak
    max_dd = float(np.max(dd)) * 100

    return {
        "symbol":         symbol,
        "n_bars":         len(df),
        "date_range":     f"{df.index[0].date()} → {df.index[-1].date()}",
        "n_sequences":    len(closed),
        "n_wins":         len(wins),
        "n_losses":       len(closed) - len(wins),
        "win_rate_pct":   len(wins) / len(profits) * 100 if profits else 0.0,
        "total_pnl_usd":  sum(profits),
        "avg_profit_usd": np.mean(profits) if profits else 0.0,
        "best_seq_usd":   max(profits) if profits else 0.0,
        "worst_seq_usd":  min(profits) if profits else 0.0,
        "max_dd_pct":     max_dd,
        "sharpe":         sharpe,
        "max_tranches":   max((len(s.tranches) for s in closed), default=0),
        "final_equity":   equity,
        "sequences":      closed,
    }


def print_results(r: dict):
    sep = "─" * 64
    print(f"\n{'═'*64}")
    print(f"  ⚡ LIMIT ORDER SCALPER (1:2 R:R) — {r['symbol']}")
    print(f"  {r['date_range']}   ({r['n_bars']:,} bars)")
    print(f"{'═'*64}")
    print(f"  Sequences completed : {r['n_sequences']}")
    print(f"  Winning sequences   : {r['n_wins']}  ({r['win_rate_pct']:.1f}%)")
    print(f"  Losing sequences    : {r['n_losses']}")
    print(f"  Total PnL           : ${r['total_pnl_usd']:>13,.2f}")
    print(f"  Avg profit/sequence : ${r['avg_profit_usd']:>13,.2f}")
    print(f"  Best sequence       : ${r['best_seq_usd']:>13,.2f}")
    print(f"  Worst sequence      : ${r['worst_seq_usd']:>13,.2f}")
    print(f"  Max drawdown        : {r['max_dd_pct']:.2f}%")
    print(f"  Sharpe ratio        : {r['sharpe']:.3f}")
    print(f"  Final equity        : ${r['final_equity']:>13,.2f}")
    print(sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",    default="XAUUSD", choices=list(DCA_CONFIG.keys()))
    ap.add_argument("--folder",    default=None)
    ap.add_argument("--parquet",   default=None)
    ap.add_argument("--csv",       default=None)
    ap.add_argument("--bars",      default=3000, type=int)
    ap.add_argument("--chunk",     default=300_000, type=int)
    ap.add_argument("--verbose",   action="store_true")
    args = ap.parse_args()

    if args.folder:
        df = load_folder_to_ohlc(args.folder, args.symbol, args.chunk)
    elif args.parquet:
        df = load_single_parquet_to_ohlc(args.parquet, args.chunk)
    elif args.csv:
        df = load_csv_to_ohlc(args.csv)
    else:
        df = generate_synthetic_ohlc(args.symbol, n_bars=args.bars)

    results = run_backtest(args.symbol, df, verbose=args.verbose)
    print_results(results)


if __name__ == "__main__":
    main()