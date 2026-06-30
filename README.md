# model-to-market-2026

The main goal for this hackathon was to see if I could abstract away all coding. I would say Claude and Gemini worked quite well, but there were times where they made mistakes that seriously hurt my pnl. Most of the code is AI generated and my input was essentially just the trading strategy.

At the end of the day, the hackathon was a great learning experience and introduction to algorithmic trading. My final strategy is based on Z-score mean reversion and basically works based on the assumption that extreme price movements in the market are temporary and over time the price of an asset will revert towards its mean. Consequently, we can place buy/sell orders based on downwards/upwards price movements and by doing this enough times make a profit.

I mostly stuck to trading on the 1 minute time frame and the strategy worked for a while until I became greedy and started shorting silver - which gave me some bigger profits until it didn't....

---
### Handoff notes for AI context across sessions

**Account:** $1,000,000 USD | Max leverage 30x | Stop-out at $0 (elimination)  
**Symbols trading:** XAUUSD, XAGUSD

## 1. How to Run

```bash
# Terminal 1 — macro agent (runs every 15 min, leave running)
python macro_regime_agent.py

# Terminal 2 — live trading engine
python main.py

# Parameter optimisation
python optimise.py

```
---
## 2. Important Files

| File | Summary |
|------|---------|
| `main.py` |  Contains trading strategy |
| `macro_regime_agent.py` |  Live web search grounded, 3-attempt retry |
| `optimise.py` | 144-parameter combination sweep + analysis w/ Claude Opus 4.8 |

---
## 3. Key Architecture

```
macro_regime_agent.py     (separate process, every 15 min)
  └── Claude + web_search → BULLISH / BEARISH / BOTH
  └── atomic write → macro_regime_state.json

main.py                   (asyncio, always running)
  ├── Task 1: tick_stream_loop × 2 symbols
  ├── Task 2: process_queue_loop
  │     └── per-symbol SymbolState
  │           └── 5s bars → z-score + tick imbalance → signal
  │                 └── position tracking → execute_order()
  ├── Task 3: risk_gate_loop (30s)
  │     └── reads macro_regime_state.json
  │     └── daily PnL → kill switch at -30 bps
  │     └── UTC midnight rollover
  └── Task 4: watchdog_loop (15s)
        └── detects disconnect per symbol
        └── reconnects with backoff (5s × attempt)
        └── halts symbol after 5 failures
        └── restarts tick_stream_loop on success
```

---
## 4. Strategy Parameters

```
entry_z           
lookback          
exit_z            
min_ticks_per_bar 
imbalance_thresh  
regime_bias       
bar_frequency     
```
---
## 5. Tech stack
- Execution: MT5 + Syphonix API
- Agents: Claude API ($50 credits)
- Schemas: Pydantic
- Backtest: DuckDB on parquet files

--- 
## 6. Dataset
- Folder: pricer-output-2026-05-11_2026-06-10/
- Structure: one parquet file per instrument per day, e.g. XAUUSD_2026_05_28.parquet
- Schema: time, sym, provider, valuedate, received, bid, ask, bidprices, askprices, bidsizes, asksizes
- 22 instruments, 27 trading days (May 11 – June 10 2026)

