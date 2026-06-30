"""
dca_long_strategy.py — Trend Pullback Scalper (LIVE ORCHESTRATION LAYER)
========================================================================
Synchronized with dca_backtest.py.
Single-bullet execution (No DCA). Tracks the "Armed" momentum state (z > 2.0).
Generates a local limit order at the 20-MA valid for 10 bars.
Calculates intra-bar High/Low boundaries directly from live tick buffers.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz

# ── Strategy Constants ────────────────────────────────────────────────────────

METALS = ["XAUUSD", "XAGUSD"]

LOOKBACK_BARS     = 20
ENTRY_Z           = 2.0
EXIT_Z            = 2.0
ARMED_WINDOW      = 10
MIN_TICKS_PER_BAR = 5
BAR_FREQ_SECONDS  = 300

DCA_CONFIG = {
    "XAUUSD": {
        "volume_base":   0.5,        # 0.5 Lots ($50/pt)
        "point_value":   100.0,
        "max_tranches":  1,          
        "profit_target": 2000.0,     # 1:2 Asymmetric R:R Target
        "emergency_sl":  -1000.0,    # Strict $1,000 Risk Limit
    },
    "XAGUSD": {
        "volume_base":   0.3,        
        "point_value":   50.0,
        "max_tranches":  1,
        "profit_target": 2000.0,      
        "emergency_sl":  -1000.0,
    },
}

STATE_FILE = "macro_regime_state.json"


# ── Per-Symbol Live State Tracking ────────────────────────────────────────────

@dataclass
class DCASymbolState:
    symbol:          str
    tick_buffer:     list        = field(default_factory=list)
    historical_bars: deque       = field(default_factory=lambda: deque(maxlen=LOOKBACK_BARS))
    historical_vols: deque       = field(default_factory=lambda: deque(maxlen=LOOKBACK_BARS))
    bar_start:       datetime    = field(default_factory=lambda: datetime.now(pytz.UTC))

    # Engine Tracking Fields
    sequence_active:     bool            = False
    tranches:            list[float]     = field(default_factory=list)
    bars_held:           int             = 0
    breakeven_locked:    bool            = False
    
    # Explicit Limit Order Target Architecture
    pending_limit_price: Optional[float] = None
    trigger_armed_bars:  int             = -1
    tp_price:            float           = 0.0
    sl_price:            float           = 0.0

    @property
    def tranche_count(self) -> int:
        return len(self.tranches)

    @property
    def total_lots(self) -> float:
        cfg = DCA_CONFIG[self.symbol]
        return cfg["volume_base"] * self.tranche_count

    @property
    def avg_entry(self) -> float:
        if not self.tranches:
            return 0.0
        return sum(self.tranches) / len(self.tranches)

    def net_pnl(self, current_price: float) -> float:
        if not self.sequence_active:
            return 0.0
        cfg = DCA_CONFIG[self.symbol]
        return self.total_lots * cfg["point_value"] * (current_price - self.avg_entry)

    def reset(self):
        self.tranches.clear()
        self.sequence_active     = False
        self.bars_held           = 0
        self.breakeven_locked    = False
        self.pending_limit_price = None
        self.trigger_armed_bars  = -1
        self.tp_price            = 0.0
        self.sl_price            = 0.0


# ── DCA Orchestrator Layer ────────────────────────────────────────────────────

class DCALongOrchestrator:
    def __init__(self, wrappers: dict, queue: asyncio.Queue):
        self.wrappers      = wrappers
        self.queue         = queue
        self.states        = {sym: DCASymbolState(sym) for sym in METALS}
        self.regime_bias   = "BOTH"
        self.total_pnl_usd = 0.0

    def _load_regime(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
            self.regime_bias = data.get("regime_bias", "BOTH")
        except Exception:
            pass

    async def _process_bar(self, sym: str):
        state = self.states[sym]
        if not state.tick_buffer:
            return

        mids        = [(t.bid + t.ask) / 2 for t in state.tick_buffer]
        close_price = mids[-1]
        h_price     = max(mids)
        l_price     = min(mids)
        tick_count  = len(mids)

        if tick_count < MIN_TICKS_PER_BAR:
            return

        state.historical_bars.append(close_price)
        state.historical_vols.append(tick_count)
        
        if len(state.historical_bars) < LOOKBACK_BARS:
            return

        series_bars = pd.Series(list(state.historical_bars))
        ma          = series_bars.mean()
        std         = series_bars.std()
        v_sma       = pd.Series(list(state.historical_vols)).mean()

        if std == 0:
            return

        z_score = (close_price - ma) / std
        self._load_regime()

        # ── 1. Live Exit Tracking Evaluation ──
        if state.sequence_active:
            state.bars_held += 1
            
            # Trailing Breakeven Optimization
            be_trigger_price = state.avg_entry + (1000.0 / (state.total_lots * DCA_CONFIG[sym]["point_value"]))
            if h_price >= be_trigger_price and not state.breakeven_locked:
                state.breakeven_locked = True
                state.sl_price = state.avg_entry

            close_reason = None
            exact_price  = close_price

            if l_price <= state.sl_price:
                close_reason = "STOP LOSS"
                exact_price  = state.sl_price
            elif h_price >= state.tp_price:
                close_reason = "TAKE PROFIT"
                exact_price  = state.tp_price
            elif ((h_price - ma) / std) >= EXIT_Z:
                close_reason = "DYNAMIC BAND EXIT"
                exact_price  = ma + (EXIT_Z * std)
            elif state.bars_held >= 48:
                close_reason = "TIME STOP"
                exact_price  = close_price

            if close_reason:
                await self._execute_sequence_close(sym, exact_price, close_reason)
                return

        # ── 2. Pending Limit Order Evaluation ──
        if not state.sequence_active and state.pending_limit_price is not None:
            if l_price <= state.pending_limit_price:
                # Execution occurs exactly at designated limit threshold
                await self._execute_sequence_open(sym, state.pending_limit_price)
                return

        # ── 3. Arming / Window Management Window ──
        if state.trigger_armed_bars >= 0:
            state.trigger_armed_bars += 1
            if state.trigger_armed_bars > ARMED_WINDOW:
                state.trigger_armed_bars  = -1
                state.pending_limit_price = None

        if z_score > ENTRY_Z and tick_count > v_sma and self.regime_bias in ("BOTH", "BULLISH"):
            state.trigger_armed_bars  = 0
            state.pending_limit_price = ma

        # Status Update
        if state.sequence_active:
            print(f"💰 [{sym}] Active | Entry: {state.avg_entry:.2f} | TP: {state.tp_price:.2f} | SL: {state.sl_price:.2f} | PnL: ${state.net_pnl(close_price):+,.2f}")
        elif state.pending_limit_price:
            print(f"⏳ [{sym}] Armed Limit Order Working @ {state.pending_limit_price:.2f} (Age: {state.trigger_armed_bars} bars)")
        else:
            print(f"📊 [{sym}] Scanning Matrix | Close: {close_price:.2f} | Z: {z_score:.2f}")

    async def _execute_sequence_open(self, sym: str, fill_price: float):
        cfg   = DCA_CONFIG[sym]
        state = self.states[sym]

        from schemas import OrderEvent, OrderAction, SignalDirection
        order = OrderEvent(
            symbol    = sym,
            action    = OrderAction.OPEN,
            direction = SignalDirection.BUY,
            volume    = cfg["volume_base"],
        )
        result = await self.wrappers[sym].execute_order(order)
        if result:
            state.tranches.append(fill_price)
            state.sequence_active = True
            
            # Derive explicit target protection bounds
            points_per_lot = cfg["volume_base"] * cfg["point_value"]
            tp_points      = cfg["profit_target"] / points_per_lot
            sl_points      = abs(cfg["emergency_sl"]) / points_per_lot
            
            state.tp_price = fill_price + tp_points
            state.sl_price = fill_price - sl_points
            
            state.pending_limit_price = None
            state.trigger_armed_bars  = -1
            print(f"📥 [{sym}] LIMIT FILLED @ {fill_price:.3f} | TP Target: {state.tp_price:.3f} | SL Floor: {state.sl_price:.3f}")

    async def _execute_sequence_close(self, sym: str, close_price: float, reason: str):
        state = self.states[sym]
        pnl   = state.net_pnl(close_price)

        from schemas import OrderEvent, OrderAction, SignalDirection
        order = OrderEvent(
            symbol    = sym,
            action    = OrderAction.CLOSE,
            direction = SignalDirection.SELL,
            volume    = state.total_lots,
        )
        result = await self.wrappers[sym].execute_order(order)
        if result:
            self.total_pnl_usd += pnl
            print(f"✅ [{sym}] SEQUENCE CLOSED ({reason}) @ {close_price:.3f} | Realized: ${pnl:,.2f} | Running Total: ${self.total_pnl_usd:,.2f}")
            state.reset()

    async def run(self):
        print("🏆 [DCA] Trend Pullback Scalper Live Layer Active.")
        while True:
            try:
                tick = await asyncio.wait_for(self.queue.get(), timeout=10.0)
                sym  = tick.sym
                if sym not in self.states:
                    self.queue.task_done()
                    continue

                state = self.states[sym]
                state.tick_buffer.append(tick)

                now = datetime.now(pytz.UTC)
                if now - state.bar_start >= timedelta(seconds=BAR_FREQ_SECONDS):
                    await self._process_bar(sym)
                    state.bar_start = now
                    state.tick_buffer.clear()

                self.queue.task_done()

            except asyncio.TimeoutError:
                now = datetime.now(pytz.UTC)
                for s in self.states.values():
                    s.bar_start = now
                    s.tick_buffer.clear()