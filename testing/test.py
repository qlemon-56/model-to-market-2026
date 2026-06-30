"""
test.py — Pre-competition validation suite
Tests all 15 competition symbols against live strategy params (Z=2.5, LB=10, SL=6bps).

Checks per symbol:
  1. Data loading      — parquet readable
  2. Spread            — median spread in bps (warn >1, fail >3)
  3. Tick density      — % of 5s bars with >= MIN_TICKS ticks
  4. Signal generation — how many Z=2.5 signals fire in one day
  5. Warm-up time      — seconds until LB=10 bars are ready

Final verdict: ranked table of all symbols by signal count + spread quality,
so you can decide which to include in main.py.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("pricer-output-2026-05-11_2026-06-10/")
con = duckdb.connect()

# ── Live strategy params — must match main.py exactly ────────────────────────
ENTRY_Z           = 2.5
LOOKBACK_BARS     = 10
MIN_TICKS_PER_BAR = 5
IMBALANCE_THRESH  = 0.1
BAR_FREQ          = "5s"

# ── All 15 competition symbols ────────────────────────────────────────────────
# Source: competition rules (BARUSD = HBAR/Hedera per Rule 21)
ALL_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "EURCHF", "EURGBP", "XAUUSD", 
    "XAGUSD", 
]

# Use a single representative day for all symbols
TEST_DATE = "2026_05_28"

PASS = "✅"
WARN = "⚠️ "
FAIL = "❌"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(symbol: str, date: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_{date}.parquet"
    df = con.execute("""
        SELECT time, bid, ask, (bid+ask)/2 AS mid, ask-bid AS spread
        FROM read_parquet($path) ORDER BY time
    """, {"path": str(path)}).df()
    df['time'] = pd.to_datetime(df['time'])
    return df

def tick_velocity_imbalance(mid_series: pd.Series) -> float:
    """Exact copy of main.py logic — must stay in sync."""
    mids = mid_series.values
    up = down = 0
    for i in range(1, len(mids)):
        if   mids[i] > mids[i-1]: up   += 1
        elif mids[i] < mids[i-1]: down += 1
    total = up + down
    return (up - down) / total if total > 0 else 0.0

def resample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('time')
    bars = df['mid'].resample(BAR_FREQ).ohlc()
    bars['spread_mean'] = df['spread'].resample(BAR_FREQ).mean()
    bars['tick_count']  = df['mid'].resample(BAR_FREQ).count()
    bars['imbalance']   = df['mid'].resample(BAR_FREQ).apply(tick_velocity_imbalance)
    return bars.dropna()

def count_signals(bars: pd.DataFrame) -> dict:
    b = bars[bars['tick_count'] >= MIN_TICKS_PER_BAR].copy()
    b['ma']  = b['close'].rolling(LOOKBACK_BARS).mean()
    b['std'] = b['close'].rolling(LOOKBACK_BARS).std()
    b['z']   = (b['close'] - b['ma']) / b['std']
    b = b.dropna()
    short = int(((b['z'] > ENTRY_Z)  & (b['imbalance'] < -IMBALANCE_THRESH)).sum())
    long  = int(((b['z'] < -ENTRY_Z) & (b['imbalance'] > IMBALANCE_THRESH)).sum())
    return {"short": short, "long": long, "total": short + long}

def warm_up_seconds(bars: pd.DataFrame) -> int:
    b = bars[bars['tick_count'] >= MIN_TICKS_PER_BAR]
    if len(b) < LOOKBACK_BARS:
        return -1
    return int((b.index[LOOKBACK_BARS - 1] - b.index[0]).total_seconds())

# ── Main validation loop ──────────────────────────────────────────────────────

print("=" * 65)
print(f"  PRE-COMPETITION VALIDATION — ALL 15 SYMBOLS")
print(f"  Date: {TEST_DATE}  |  Z={ENTRY_Z}  LB={LOOKBACK_BARS}  MinTicks={MIN_TICKS_PER_BAR}")
print("=" * 65)

summary = []  # For ranked table at end

for symbol in ALL_SYMBOLS:
    print(f"\n── {symbol} ──────────────────────────────────────────")

    row = {"symbol": symbol, "loaded": False, "med_spread": None,
           "tick_pct": None, "signals": 0, "warmup": None, "verdict": FAIL}

    # Check 1: Data loading
    try:
        df = load(symbol, TEST_DATE)
        row["loaded"] = True
        print(f"  {PASS} Data: {len(df):,} ticks")
    except Exception as e:
        print(f"  {FAIL} Could not load: {e}")
        summary.append(row)
        continue

    # Check 2: Spread
    spread_bps = df['spread'] / df['bid'] * 10000
    med = float(spread_bps.median())
    row["med_spread"] = med
    s_flag = PASS if med < 1.0 else (WARN if med < 3.0 else FAIL)
    print(f"  {s_flag} Spread: median={med:.3f} bps  p99={spread_bps.quantile(0.99):.2f} bps")

    # Check 3: Tick density
    bars = resample(df)
    pct = float((bars['tick_count'] >= MIN_TICKS_PER_BAR).mean() * 100)
    avg = float(bars['tick_count'].mean())
    row["tick_pct"] = pct
    t_flag = PASS if pct >= 80 else (WARN if pct >= 50 else FAIL)
    print(f"  {t_flag} Tick density: {pct:.1f}% of bars >= {MIN_TICKS_PER_BAR} ticks  (avg {avg:.1f}/bar)")

    # Check 4: Signal generation
    sigs = count_signals(bars)
    row["signals"] = sigs["total"]
    sig_flag = PASS if sigs["total"] >= 3 else (WARN if sigs["total"] >= 1 else FAIL)
    print(f"  {sig_flag} Signals at Z={ENTRY_Z}: {sigs['total']} total  "
          f"(short={sigs['short']}  long={sigs['long']})")

    # Check 5: Warm-up
    wu = warm_up_seconds(bars)
    row["warmup"] = wu
    w_flag = PASS if 0 < wu <= 120 else WARN
    print(f"  {w_flag} Warm-up: {wu}s until {LOOKBACK_BARS} bars ready")

    # Per-symbol verdict
    if med < 3.0 and pct >= 50 and sigs["total"] >= 1:
        row["verdict"] = PASS if (med < 1.0 and pct >= 80 and sigs["total"] >= 3) else WARN
    else:
        row["verdict"] = FAIL

    summary.append(row)

# ── Ranked summary table ──────────────────────────────────────────────────────

print(f"\n\n{'=' * 65}")
print(f"  RANKED SUMMARY — sorted by signal count then spread")
print(f"{'=' * 65}")
print(f"  {'Symbol':<8}  {'Spread':>8}  {'TickPct':>8}  {'Signals':>8}  {'Warmup':>7}  {'Verdict'}")
print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

df_summary = pd.DataFrame(summary)
df_loaded  = df_summary[df_summary['loaded']].copy()
df_failed  = df_summary[~df_summary['loaded']].copy()

# Sort: verdict PASS first, then by signals desc, then spread asc
verdict_order = {PASS: 0, WARN: 1, FAIL: 2}
df_loaded['sort_verdict'] = df_loaded['verdict'].map(verdict_order)
df_loaded = df_loaded.sort_values(['sort_verdict', 'signals', 'med_spread'],
                                   ascending=[True, False, True])

for _, r in df_loaded.iterrows():
    spread_str = f"{r['med_spread']:.3f}" if r['med_spread'] is not None else "N/A"
    tick_str   = f"{r['tick_pct']:.1f}%" if r['tick_pct'] is not None else "N/A"
    warmup_str = f"{r['warmup']}s"        if r['warmup'] is not None else "N/A"
    print(f"  {r['symbol']:<8}  {spread_str:>8}  {tick_str:>8}  {r['signals']:>8}  {warmup_str:>7}  {r['verdict']}")

for _, r in df_failed.iterrows():
    print(f"  {r['symbol']:<8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>7}  {FAIL} (no data)")

# ── Recommendation ────────────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
recommended = df_loaded[df_loaded['verdict'].isin([PASS, WARN])]['symbol'].tolist()
passed      = df_loaded[df_loaded['verdict'] == PASS]['symbol'].tolist()
warned      = df_loaded[df_loaded['verdict'] == WARN]['symbol'].tolist()
failed_syms = df_loaded[df_loaded['verdict'] == FAIL]['symbol'].tolist() + df_failed['symbol'].tolist()

print(f"  {PASS} TRADE ({len(passed)}):   {', '.join(passed) if passed else 'None'}")
print(f"  {WARN} MONITOR ({len(warned)}): {', '.join(warned) if warned else 'None'}")
print(f"  {FAIL} SKIP ({len(failed_syms)}):    {', '.join(failed_syms) if failed_syms else 'None'}")

if recommended:
    syms_str = '", "'.join(recommended)
    print(f"\n  Suggested SYMBOLS list for main.py:")
    print(f'  SYMBOLS = ["{syms_str}"]')

print("=" * 65)