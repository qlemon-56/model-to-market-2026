import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time

# ─── MULTI-ASSET CONFIGURATION MATRIX ──────────────────────────────────
CONFIG = {
    "XAUUSD": {
        "LOT_SIZE": 60,
        "MAX_HOLD_MINUTES": 15,     
        "EARLY_PROFIT_MOVE": 1.0,    
        "MAGIC_NUMBER": 777888
    },
    "XAGUSD": {
        "LOT_SIZE": 60,
        "MAX_HOLD_MINUTES": 15,     
        "EARLY_PROFIT_MOVE": 0.10,    
        "MAGIC_NUMBER": 777999
    }
}

LOOKBACK = 20
ENTRY_Z = 2.2                       

# TRACKING MEMORY: Neutralizes same-candle duplicate trading loops
LAST_TRADED_BAR = {symbol: None for symbol in CONFIG}

def initialize_mt5():
    """Initializes the connection to the active MetaTrader 5 terminal environment."""
    if not mt5.initialize():
        print(f"❌ Initialize failed. Error: {mt5.last_error()}")
        quit()
    if not mt5.terminal_info().trade_allowed:
        print("⚠️ WARNING: 'Algo Trading' is disabled in your MT5 terminal interface. Please enable it.")
    print("🚀 Connected to MT5 Live Market Multi-Pipeline Engine.")

def get_market_data(symbol, timeframe, num_candles=100):
    """Dynamically parses and builds standard deviation distributions for signals."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['mean'] = df['close'].rolling(LOOKBACK).mean()
    df['std'] = df['close'].rolling(LOOKBACK).std()
    
    # FIX: Floor dropped from 0.15 to 0.001 so Silver's micro-deviations are no longer muted
    df['z_score'] = (df['close'] - df['mean']) / np.maximum(df['std'], 0.001)
    return df

def get_open_position(symbol, magic_number):
    """Queries active terminal inventory using strict strategy magic identifier separation."""
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            if pos.magic == magic_number:
                return pos
    return None

def get_filling_mode(symbol):
    """Queries broker protocol structures to guarantee immediate order execution."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return mt5.ORDER_FILLING_IOC
        
    filling_mask = symbol_info.filling_mode
    if filling_mask & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling_mask & 2:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

def execute_trade(action, symbol, lot, price, magic_number, position_ticket=None):
    """High-reliability transaction handler utilizing a 50-point live slippage buffer."""
    order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
    filling_mode = get_filling_mode(symbol)
    
    request = {
        "action": int(mt5.TRADE_ACTION_DEAL),
        "symbol": str(symbol),
        "volume": float(lot),
        "type": int(order_type),
        "price": float(price),
        "deviation": int(50),  
        "magic": int(magic_number),
        "comment": f"Z-Scalper {symbol}",
        "type_time": int(mt5.ORDER_TIME_GTC),
        "type_filling": int(filling_mode),
    }
    
    if position_ticket:
        request["position"] = int(position_ticket)
        request["comment"] = f"Z-Exit {symbol}"

    result = mt5.order_send(request)
    
    if result is None:
        print(f"❌ Python Core API Error: {mt5.last_error()}")
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Broker execution rejected for {symbol}. Code: {result.retcode} | {result.comment}")
    else:
        action_name = "OPENED" if not position_ticket else "CLOSED"
        print(f"\n✅ [{symbol}] TRADE {action_name} EXECUTED. Ticket ID: {result.order} @ {result.price}")

def run_strategy():
    """Execution supervisor scanning across isolated data tracks concurrently."""
    print("🔥 Multi-Asset Z-Score Engine Online. Scanning tracks...")
    for symbol in CONFIG:
        mt5.symbol_select(symbol, True)

    while True:
        status_updates = []
        
        for symbol, settings in CONFIG.items():
            df = get_market_data(symbol, mt5.TIMEFRAME_M1)
            tick = mt5.symbol_info_tick(symbol)
            
            if df is None or len(df) < (LOOKBACK + 2) or tick is None:
                continue

            current_bar_time = df.iloc[-2]['time']
            current_close = df.iloc[-2]['close']
            current_mean = df.iloc[-2]['mean']
            current_z = df.iloc[-2]['z_score']
            
            pos = get_open_position(symbol, settings["MAGIC_NUMBER"])

            # ─── 1. DYNAMIC EXITS & AUTOMATIC EXPIRATIONS ───────────────────
            if pos:
                trade_duration_mins = (tick.time - pos.time) / 60.0
                is_stale = trade_duration_mins >= settings["MAX_HOLD_MINUTES"]

                open_profit_price = 0.0
                if pos.type == mt5.POSITION_TYPE_BUY:
                    open_profit_price = tick.bid - pos.price_open
                elif pos.type == mt5.POSITION_TYPE_SELL:
                    open_profit_price = pos.price_open - tick.ask

                early_profit_hit = open_profit_price >= settings["EARLY_PROFIT_MOVE"]

                if pos.type == mt5.POSITION_TYPE_BUY:
                    if current_close >= current_mean or is_stale or early_profit_hit:
                        reason = "PROFIT TARGET SECURED" if early_profit_hit else ("TIMEOUT ENFORCED" if is_stale else "MEAN RETRACEMENT")
                        print(f"\n<<< EXITING LONG [{symbol}] ({reason} | Duration: {trade_duration_mins:.1f}m) >>>")
                        execute_trade('SELL', symbol, pos.volume, tick.bid, settings["MAGIC_NUMBER"], pos.ticket)
                        
                elif pos.type == mt5.POSITION_TYPE_SELL:
                    if current_close <= current_mean or is_stale or early_profit_hit:
                        reason = "PROFIT TARGET SECURED" if early_profit_hit else ("TIMEOUT ENFORCED" if is_stale else "MEAN RETRACEMENT")
                        print(f"\n<<< EXITING SHORT [{symbol}] ({reason} | Duration: {trade_duration_mins:.1f}m) >>>")
                        execute_trade('BUY', symbol, pos.volume, tick.ask, settings["MAGIC_NUMBER"], pos.ticket)

                status_updates.append(f"{symbol}: ACTIVE ({trade_duration_mins:.1f}m)")
                continue  

            # ─── 2. ENTRY LOGIC PROCESSED WHEN FLAT ─────────────────────────
            status_updates.append(f"{symbol}: Z {current_z:+.2f}")

            if LAST_TRADED_BAR[symbol] == current_bar_time:
                continue

            if current_z <= -ENTRY_Z:
                print(f"\n>>> BUY SIGNAL [{symbol}]: Velocity Extension Identified (Z: {current_z:+.2f}) <<<")
                execute_trade('BUY', symbol, settings["LOT_SIZE"], tick.ask, settings["MAGIC_NUMBER"])
                LAST_TRADED_BAR[symbol] = current_bar_time  
                time.sleep(1) 
                
            elif current_z >= ENTRY_Z:
                print(f"\n>>> SELL SIGNAL [{symbol}]: Velocity Extension Identified (Z: {current_z:+.2f}) <<<")
                execute_trade('SELL', symbol, settings["LOT_SIZE"], tick.bid, settings["MAGIC_NUMBER"])
                LAST_TRADED_BAR[symbol] = current_bar_time  
                time.sleep(1) 

        print(f"Scanning | {' | '.join(status_updates)}          ", end="\r", flush=True)
        time.sleep(1)

if __name__ == "__main__":
    initialize_mt5()
    try:
        run_strategy()
    except KeyboardInterrupt:
        print("\nHalting multi-asset tracking session safely...")
    finally:
        mt5.shutdown()