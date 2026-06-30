"""
optimise.py — Parameter sweep + Claude-powered analysis for the mean reversion strategy.

Architecture:
  1. Load all parquet days once into memory (avoid repeated disk I/O per combo)
  2. Pre-compute resampled bars for each day (tick imbalance is expensive — do it once)
  3. Run a vectorised parameter grid sweep across all combinations
  4. Rank by a composite score: Sharpe * profit_factor (rewards consistency + edge together)
  5. Feed the top-10 results + worst-10 to Claude Sonnet for qualitative analysis
     and a recommended parameter set with reasoning
  6. Print the full ranked table + Claude's recommendation

Parameter space (144 combinations):
  stop_loss_bps      : [6, 8, 10, 12]
  entry_z            : [1.8, 2.0, 2.2, 2.5]
  min_ticks_per_bar  : [3, 5, 10]          — filters thin/illiquid bars
  lookback           : [10, 15, 20]
"""

import duckdb
import pandas as pd
import numpy as np
import itertools
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

DATA_DIR = Path("pricer-output-2026-05-11_2026-06-10/")
con = duckdb.connect()

INSTRUMENT = "XAUUSD"
DATES = [
    "2026_05_19", "2026_05_20", "2026_05_21", "2026_05_22",
    "2026_05_25", "2026_05_26", "2026_05_27", "2026_05_28",
    "2026_06_01", "2026_06_02", "2026_06_03", "2026_06_04",
]

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SweepParams(BaseModel):
    """A single parameter combination to evaluate."""
    stop_loss_bps:     float = Field(..., ge=1.0,  le=50.0)
    entry_z:           float = Field(..., ge=0.5,  le=5.0)
    min_ticks_per_bar: int   = Field(..., ge=1,    le=100)
    lookback:          int   = Field(..., ge=5,    le=100)
    exit_z:            float = Field(default=0.3)
    spread_filter_bps: float = Field(default=0.8)
    imbalance_threshold: float = Field(default=0.1)
    regime_bias:       str   = Field(default="BEARISH")

    @field_validator("regime_bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        if v not in {"BULLISH", "BEARISH", "BOTH"}:
            raise ValueError(f"Invalid regime_bias: {v}")
        return v

class SweepResult(BaseModel):
    """Metrics for a single parameter combination after running the full backtest."""
    params:          SweepParams
    total_trades:    int
    win_rate:        float
    avg_pnl:         float
    total_pnl:       float
    std_pnl:         float
    sharpe:          float
    profit_factor:   float
    max_loss:        float
    sl_hits:         int
    avg_bars_held:   float
    composite_score: float   # sharpe * profit_factor — primary ranking metric

# ── Data loading ──────────────────────────────────────────────────────────────

def load_day(date: str) -> pd.DataFrame:
    path = DATA_DIR / f"{INSTRUMENT}_{date}.parquet"
    df = con.execute("""
        SELECT time, bid, ask, (bid+ask)/2 AS mid, ask-bid AS spread
        FROM read_parquet($path) ORDER BY time
    """, {"path": str(path)}).df()
    df['time'] = pd.to_datetime(df['time'])
    return df

# ── Tick velocity imbalance ───────────────────────────────────────────────────

def compute_tick_velocity_imbalance(mid_series: pd.Series) -> float:
    mids = mid_series.values
    up = down = 0
    for i in range(1, len(mids)):
        if   mids[i] > mids[i-1]: up   += 1
        elif mids[i] < mids[i-1]: down += 1
    total = up + down
    return (up - down) / total if total > 0 else 0.0

# ── Bar resampling (done once per day, shared across all param combos) ────────

def resample_to_bars(df: pd.DataFrame, freq: str = '5s') -> pd.DataFrame:
    df = df.set_index('time')
    bars = df['mid'].resample(freq).ohlc()
    bars['spread_mean'] = df['spread'].resample(freq).mean()
    bars['tick_count']  = df['mid'].resample(freq).count()
    bars['imbalance']   = df['mid'].resample(freq).apply(compute_tick_velocity_imbalance)
    return bars.dropna()

# ── Core backtest (runs on pre-computed bars) ─────────────────────────────────

def run_backtest(bars_list: list[pd.DataFrame], p: SweepParams) -> list[dict]:
    """Run the strategy across all pre-computed daily bar frames with params p."""
    trades: list[dict] = []

    # Per-symbol std floor: ~1.5 bps of XAUUSD typical mid price.
    # Must match backtest.py: mid_price_for_std_floor * 0.000015
    # Optimiser runs XAUUSD only (mid ~3300), so floor = 3300 * 0.000015 = 0.0495
    MIN_STD_FLOOR = 3300.0 * 0.000015  # ~0.0495

    for bars in bars_list:
        b = bars.copy()

        # Z-score
        b['mid_ma']  = b['close'].rolling(p.lookback).mean()
        b['mid_std'] = b['close'].rolling(p.lookback).std()
        b['z']       = (b['close'] - b['mid_ma']) / np.maximum(b['mid_std'], MIN_STD_FLOOR)

        # Filters — spread + tick density
        b['spread_bps'] = b['spread_mean'] / b['close'] * 10000
        b['tradeable']  = (b['spread_bps'] < p.spread_filter_bps) & (b['tick_count'] >= p.min_ticks_per_bar)

        # Volatility gate: block ghost signals during low-vol consolidation (aligned with backtest.py)
        b['vol_ok'] = b['mid_std'] >= MIN_STD_FLOOR

        position    = 0
        entry_price = 0.0
        entry_bar   = 0
        rows        = list(b.iterrows())

        for i, (ts, row) in enumerate(rows):
            if not row['tradeable'] or pd.isna(row['z']) or not row['vol_ok']:
                continue

            # Exit long
            if position == 1:
                pnl = (row['close'] - entry_price) / entry_price * 10000
                if pnl <= -p.stop_loss_bps:
                    trades.append({'pnl_bps': -p.stop_loss_bps, 'bars_held': i - entry_bar, 'exit_reason': 'SL'})
                    position = 0
                elif row['z'] > -p.exit_z:
                    trades.append({'pnl_bps': pnl, 'bars_held': i - entry_bar, 'exit_reason': 'Target'})
                    position = 0

            # Exit short
            elif position == -1:
                pnl = (entry_price - row['close']) / entry_price * 10000
                if pnl <= -p.stop_loss_bps:
                    trades.append({'pnl_bps': -p.stop_loss_bps, 'bars_held': i - entry_bar, 'exit_reason': 'SL'})
                    position = 0
                elif row['z'] < p.exit_z:
                    trades.append({'pnl_bps': pnl, 'bars_held': i - entry_bar, 'exit_reason': 'Target'})
                    position = 0

            # Entry
            if position == 0:
                if row['z'] > p.entry_z and row['imbalance'] < -p.imbalance_threshold:
                    if p.regime_bias in ["BOTH", "BEARISH"]:
                        position = -1; entry_price = row['close']; entry_bar = i
                elif row['z'] < -p.entry_z and row['imbalance'] > p.imbalance_threshold:
                    if p.regime_bias in ["BOTH", "BULLISH"]:
                        position = 1; entry_price = row['close']; entry_bar = i

    return trades

# ── Metrics calculator ────────────────────────────────────────────────────────

def compute_metrics(trades: list[dict], p: SweepParams) -> SweepResult:
    if len(trades) < 5:
        return SweepResult(
            params=p, total_trades=len(trades), win_rate=0, avg_pnl=0,
            total_pnl=0, std_pnl=0, sharpe=-99, profit_factor=0,
            max_loss=0, sl_hits=0, avg_bars_held=0, composite_score=-99
        )
    pnls  = pd.Series([t['pnl_bps'] for t in trades])
    wins  = pnls[pnls > 0].sum()
    losses= pnls[pnls < 0].sum()
    pf    = wins / abs(losses) if losses != 0 else float('inf')
    std   = pnls.std()
    sharpe= pnls.mean() / std if std > 0 else 0.0
    comp  = sharpe * pf

    return SweepResult(
        params=p,
        total_trades=len(trades),
        win_rate=float((pnls > 0).mean()),
        avg_pnl=float(pnls.mean()),
        total_pnl=float(pnls.sum()),
        std_pnl=float(std),
        sharpe=float(sharpe),
        profit_factor=float(pf),
        max_loss=float(pnls.min()),
        sl_hits=int(sum(1 for t in trades if t['exit_reason'] == 'SL')),
        avg_bars_held=float(np.mean([t['bars_held'] for t in trades])),
        composite_score=float(comp),
    )

# ── Claude analysis ───────────────────────────────────────────────────────────

def analyse_with_claude(ranked: list[SweepResult]) -> str:
    """Send top-10 and bottom-10 results to Claude for qualitative analysis."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️  ANTHROPIC_API_KEY not set — skipping Claude analysis."

    client = anthropic.Anthropic(api_key=api_key)

    def result_to_dict(r: SweepResult) -> dict[str, Any]:
        return {
            "stop_loss_bps":     r.params.stop_loss_bps,
            "entry_z":           r.params.entry_z,
            "min_ticks_per_bar": r.params.min_ticks_per_bar,
            "lookback":          r.params.lookback,
            "total_trades":      r.total_trades,
            "win_rate":          round(r.win_rate, 3),
            "total_pnl":         round(r.total_pnl, 2),
            "sharpe":            round(r.sharpe, 4),
            "profit_factor":     round(r.profit_factor, 3),
            "sl_hits":           r.sl_hits,
            "composite_score":   round(r.composite_score, 4),
        }

    top10    = [result_to_dict(r) for r in ranked[:10]]
    bottom10 = [result_to_dict(r) for r in ranked[-10:]]

    prompt = f"""You are a quantitative trading analyst reviewing parameter sweep results 
for a mean-reversion strategy on XAUUSD (Gold CFD) using 5-second bars.

The strategy:
- Enters short when z-score > entry_z AND tick velocity imbalance < -0.1 (bearish regime only)
- Exits on z-score reversion to ±0.3 OR hard stop-loss
- Composite score = Sharpe × Profit Factor (primary ranking metric)
- Tested on 12 trading days (May–June 2026), ~272 baseline trades

TOP 10 parameter combinations (by composite score):
{json.dumps(top10, indent=2)}

BOTTOM 10 parameter combinations:
{json.dumps(bottom10, indent=2)}

Please provide:
1. What parameter patterns separate the top performers from the bottom performers?
2. Which single parameter has the most impact on Sharpe and why?
3. Your recommended parameter set (must be one of the top 10) with justification.
4. Any overfitting risks to flag — are the top results suspiciously good?
5. One specific thing to test next to further improve the edge.

Be concise and direct. Focus on actionable insights."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )

    text = ""
    for block in response.content:
        text_content = getattr(block, "text", "")
        if text_content:
            text = text_content
            break
    return text

# ── Main sweep ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🔄 Loading and resampling {len(DATES)} days of tick data...")
    bars_list: list[pd.DataFrame] = []
    for date in DATES:
        try:
            df   = load_day(date)
            bars = resample_to_bars(df, freq='5min')
            bars_list.append(bars)
            print(f"  ✅ {date}  ({len(bars)} bars)")
        except Exception as e:
            print(f"  ⚠️  {date} SKIP — {e}")

    print(f"\n📐 Bars loaded for {len(bars_list)} days. Starting parameter sweep...\n")

    # Parameter grid
    grid = list(itertools.product(
        [6.0, 8.0, 10.0, 12.0],   # stop_loss_bps
        [1.8, 2.0, 2.2, 2.5],     # entry_z
        [3, 5, 10],                # min_ticks_per_bar
        [10, 15, 20],              # lookback
    ))
    print(f"🧪 {len(grid)} combinations to evaluate...\n")

    results: list[SweepResult] = []
    for i, (sl, ez, mt, lb) in enumerate(grid):
        p = SweepParams(
            stop_loss_bps=sl,
            entry_z=ez,
            min_ticks_per_bar=mt,
            lookback=lb,
        )
        trades  = run_backtest(bars_list, p)
        metrics = compute_metrics(trades, p)
        results.append(metrics)

        if (i + 1) % 24 == 0:
            print(f"  ... {i+1}/{len(grid)} done")

    # Rank by composite score
    ranked = sorted(results, key=lambda r: r.composite_score, reverse=True)

    # ── Print full ranked table ───────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  FULL SWEEP RESULTS  ({len(ranked)} combinations, ranked by Sharpe × Profit Factor)")
    print(f"{'='*95}")
    header = f"{'#':>3}  {'SL':>5}  {'Z':>5}  {'MinTk':>5}  {'LB':>4}  {'Trades':>6}  {'WR':>6}  {'PnL':>8}  {'Sharpe':>7}  {'PF':>6}  {'SLhits':>6}  {'Score':>7}"
    print(header)
    print("-" * 95)
    for rank, r in enumerate(ranked, 1):
        p = r.params
        flag = " ⭐" if rank <= 3 else ""
        print(
            f"{rank:>3}  {p.stop_loss_bps:>5.1f}  {p.entry_z:>5.2f}  {p.min_ticks_per_bar:>5d}  "
            f"{p.lookback:>4d}  {r.total_trades:>6d}  {r.win_rate:>6.1%}  "
            f"{r.total_pnl:>8.2f}  {r.sharpe:>7.4f}  {r.profit_factor:>6.3f}  "
            f"{r.sl_hits:>6d}  {r.composite_score:>7.4f}{flag}"
        )

    # ── Top 5 summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TOP 5 COMBINATIONS")
    print(f"{'='*60}")
    for rank, r in enumerate(ranked[:5], 1):
        p = r.params
        print(f"\n  #{rank}  SL={p.stop_loss_bps}bps  Z={p.entry_z}  MinTicks={p.min_ticks_per_bar}  LB={p.lookback}")
        print(f"       Trades={r.total_trades}  WR={r.win_rate:.1%}  PnL={r.total_pnl:.2f}bps")
        print(f"       Sharpe={r.sharpe:.4f}  PF={r.profit_factor:.3f}  SL hits={r.sl_hits}")
        print(f"       Composite={r.composite_score:.4f}")

    # ── Baseline for reference ────────────────────────────────────────────────
    baseline = next((r for r in ranked if r.params.stop_loss_bps == 12.0
                     and r.params.entry_z == 2.0
                     and r.params.min_ticks_per_bar == 3
                     and r.params.lookback == 20), None)
    if baseline:
        print(f"\n  BASELINE (SL=12, Z=2.0, MinTicks=3, LB=20):")
        print(f"  Rank #{ranked.index(baseline)+1}  Sharpe={baseline.sharpe:.4f}  Score={baseline.composite_score:.4f}")

    # ── Claude analysis ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🧠 CLAUDE ANALYSIS")
    print(f"{'='*60}\n")
    analysis = analyse_with_claude(ranked)
    print(analysis)

    print(f"\n✅ Sweep complete at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")