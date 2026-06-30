"""
backtest_multisymbol.py — Multi-symbol diagnostic backtest (5-min bars).

Runs 4 variants across all 10 FX/metals symbols to compare regime,
Z-threshold, and per-symbol contribution to portfolio PnL.

Bar frequency: 5 minutes (aligned with live main.py)
max_hold_bars: 12 bars = 60 minutes max hold at 5-min granularity
                (was 60 bars = 5 min at 5s — rescaled 1:1 in time)

  A. BOTH regime  — baseline, no directional filter
  B. BEARISH only — original validated regime
  C. BEARISH + tighter Z=2.5
  D. BEARISH + Z=2.5 + drop XAGUSD (high SL rate)
"""

import pandas as pd
import numpy as np
from backtest import load_day, resample_to_bars, backtest_mean_reversion

# ── Config mirroring main.py ──────────────────────────────────────────────────

SYMBOLS = [
    "USDJPY", "USDCAD", "EURUSD", "GBPUSD", "AUDUSD",
    "EURGBP", "XAUUSD", "USDCHF", "XAGUSD", "EURCHF",
]

TYPICAL_PRICE = {
    "USDJPY": 155.0, "USDCAD": 1.38,  "EURUSD": 1.08,
    "GBPUSD": 1.27,  "AUDUSD": 0.64,  "EURGBP": 0.85,
    "XAUUSD": 3300.0,"USDCHF": 0.90,  "XAGUSD": 33.0,
    "EURCHF": 0.94,
}

STOP_LOSS_POINTS = {
    "USDJPY": 0.15,     "USDCAD": 0.00138, "EURUSD": 0.00108,
    "GBPUSD": 0.00127,  "AUDUSD": 0.00064, "EURGBP": 0.00085,
    "XAUUSD": 3.30,     "USDCHF": 0.00090, "XAGUSD": 0.033,
    "EURCHF": 0.00094,
}

STOP_LOSS_BPS = {
    sym: (STOP_LOSS_POINTS[sym] / TYPICAL_PRICE[sym]) * 10000
    for sym in SYMBOLS
}

MIN_STD_FLOOR = {
    "USDJPY": 0.0023,    "USDCAD": 0.000021,
    "EURUSD": 0.000016,  "GBPUSD": 0.000019,
    "AUDUSD": 0.0000096, "EURGBP": 0.0000128,
    "XAUUSD": 0.050,     "USDCHF": 0.0000135,
    "XAGUSD": 0.0005,    "EURCHF": 0.0000141,
}

DATES = [
    "2026_05_19", "2026_05_20", "2026_05_21", "2026_05_22",
    "2026_05_25", "2026_05_26", "2026_05_27", "2026_05_28",
    "2026_06_01", "2026_06_02", "2026_06_03", "2026_06_04",
]

# ── Variants to test ──────────────────────────────────────────────────────────

VARIANTS = [
    {
        "label":        "A — BOTH regime (no directional filter, baseline)",
        "regime":       "BOTH",
        "spread_filt":  0.8,
        "symbols":      SYMBOLS,
        "entry_z":      2.0,
    },
    {
        "label":        "B — BEARISH only (validated regime)",
        "regime":       "BEARISH",
        "spread_filt":  0.8,
        "symbols":      SYMBOLS,
        "entry_z":      2.0,
    },
    {
        "label":        "C — BEARISH + tighter Z=2.5",
        "regime":       "BEARISH",
        "spread_filt":  0.8,
        "symbols":      SYMBOLS,
        "entry_z":      2.5,
    },
    {
        "label":        "D — BEARISH + Z=2.5 + drop XAGUSD (high SL rate)",
        "regime":       "BEARISH",
        "spread_filt":  0.8,
        "symbols":      [s for s in SYMBOLS if s != "XAGUSD"],
        "entry_z":      2.5,
    },
]

# ── Runner ────────────────────────────────────────────────────────────────────

def run_variant(variant: dict) -> pd.DataFrame:
    all_trades = []
    for sym in variant["symbols"]:
        sl_bps    = STOP_LOSS_BPS[sym]
        mid_price = TYPICAL_PRICE[sym]
        spread_f  = variant["spread_filt"]
        # XAGUSD needs a wider spread filter regardless
        if sym == "XAGUSD":
            spread_f = 2.5

        for date in DATES:
            try:
                df   = load_day(sym, date)
                bars = resample_to_bars(df, freq='5min')
                trades = backtest_mean_reversion(
                    bars,
                    entry_z=variant["entry_z"],
                    lookback=20,
                    stop_loss_bps=sl_bps,
                    spread_filter_bps=spread_f,
                    imbalance_threshold=0.1,
                    regime_bias=variant["regime"],
                    use_session_filter=False,
                    max_daily_loss_bps=30.0,
                    max_hold_bars=12,          # 12 × 5min = 60 min max hold
                    min_ticks_per_bar=5,
                    mid_price_for_std_floor=mid_price,
                )
                if len(trades) > 0:
                    trades['date']   = date
                    trades['symbol'] = sym
                    all_trades.append(trades)
            except Exception as e:
                pass  # Skip missing data silently

    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def print_variant(df: pd.DataFrame, label: str):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    if df.empty:
        print("  No trades.")
        return

    pnl    = df['pnl_bps']
    wins   = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    pf     = wins / abs(losses) if losses != 0 else float('inf')
    sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
    sl_hits = (df['exit_reason'] == 'SL').sum()

    # 15-min Sharpe
    tmp = df.copy()
    tmp['ts'] = pd.to_datetime(tmp['ts'])
    wp  = tmp.set_index('ts')['pnl_bps'].resample('15min').sum().fillna(0)
    s15 = wp.mean() / wp.std() if wp.std() > 0 else 0

    print(f"  Trades:          {len(df)}")
    print(f"  Win rate:        {pnl.gt(0).mean():.1%}")
    print(f"  Total PnL:       {pnl.sum():+.2f} bps")
    print(f"  Avg PnL/trade:   {pnl.mean():+.4f} bps")
    print(f"  Trade Sharpe:    {sharpe:.4f}")
    print(f"  15-min Sharpe:   {s15:.4f}  ← competition metric")
    print(f"  Profit factor:   {pf:.3f}")
    print(f"  SL hits:         {sl_hits} ({sl_hits/len(df):.1%})")

    # Per-symbol breakdown
    print(f"\n  Per-symbol breakdown:")
    print(f"  {'Sym':<8} {'Trades':>6} {'WR':>6} {'PnL':>9} {'Shr':>7} {'SLhits':>6}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*9} {'-'*7} {'-'*6}")

    sym_stats = []
    for sym in df['symbol'].unique():
        s = df[df['symbol'] == sym]['pnl_bps']
        sh = s.mean() / s.std() if s.std() > 0 else 0
        sl = (df[(df['symbol']==sym)]['exit_reason'] == 'SL').sum()
        sym_stats.append((sh, sym, len(s), s.gt(0).mean(), s.sum(), sh, sl))

    for _, sym, n, wr, tot, sh, sl in sorted(sym_stats, reverse=True):
        flag = "✅" if sh > 0 else "❌"
        print(f"  {sym:<8} {n:>6} {wr:>6.1%} {tot:>+9.2f} {sh:>7.4f} {sl:>6}  {flag}")

    # Daily portfolio PnL
    print(f"\n  Daily PnL:")
    tmp2 = df.copy()
    tmp2['ts'] = pd.to_datetime(tmp2['ts'])
    tmp2['day'] = tmp2['ts'].dt.date
    daily = tmp2.groupby('day')['pnl_bps'].sum()
    for day, val in daily.items():
        bar  = '█' * min(int(abs(val) / 10), 30)
        sign = '+' if val >= 0 else '-'
        print(f"    {day}  {sign}{abs(val):7.2f} bps  {bar}")
    print(f"\n    TOTAL: {pnl.sum():+.2f} bps")


# ── Run all variants ──────────────────────────────────────────────────────────

print("=" * 65)
print("  MULTI-SYMBOL DIAGNOSTIC BACKTEST  (5-min bars)")
print("  10 symbols × 4 variants — regime / Z / symbol attribution")
print("=" * 65)

results = {}
for v in VARIANTS:
    print(f"\n⏳ Running: {v['label']}...")
    df = run_variant(v)
    results[v['label']] = df
    print_variant(df, v['label'])

# ── Summary comparison ────────────────────────────────────────────────────────

print(f"\n\n{'='*65}")
print(f"  VARIANT COMPARISON SUMMARY")
print(f"{'='*65}")
print(f"  {'Variant':<45} {'Trades':>6} {'PnL':>9} {'15mShr':>8}")
print(f"  {'-'*45} {'-'*6} {'-'*9} {'-'*8}")

for v in VARIANTS:
    label = v['label']
    df    = results[label]
    if df.empty:
        print(f"  {label[:45]:<45} {'—':>6} {'—':>9} {'—':>8}")
        continue
    pnl = df['pnl_bps']
    tmp = df.copy()
    tmp['ts'] = pd.to_datetime(tmp['ts'])
    wp  = tmp.set_index('ts')['pnl_bps'].resample('15min').sum().fillna(0)
    s15 = wp.mean() / wp.std() if wp.std() > 0 else 0
    print(f"  {label[:45]:<45} {len(df):>6} {pnl.sum():>+9.2f} {s15:>8.4f}")

print(f"\n  Key insight: compare A vs B to see regime impact,")
print(f"               B vs C to see Z threshold impact,")
print(f"               C vs D to see XAGUSD impact.")