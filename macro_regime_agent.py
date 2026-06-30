import time
import json
import os
from datetime import datetime, UTC
import anthropic
from dotenv import load_dotenv

load_dotenv()

STATE_FILE = "macro_regime_state.json"
UPDATE_INTERVAL_SECONDS = 900  # 15 minutes
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries


class MacroRegimeAgent:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("🚨 ANTHROPIC_API_KEY not found in environment or .env file!")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.current_regime = "BOTH"
    def analyze_market_regime(self) -> str:
        """
        Calls Claude with web_search enabled so it can retrieve live Gold, DXY,
        and rates data before making a regime decision.
        Retries up to MAX_RETRIES times before defaulting to BOTH.
        """
        prompt = """
        You are a quantitative macro researcher. Use the web_search tool to look up the 
        following RIGHT NOW before answering:

        1. US Dollar Index (DXY) current level, today's % move, and short-term trend (rising/falling/flat)
        2. US 10-year Treasury yield current level and intraday direction
        3. Current Gold (XAUUSD) spot price, today's move, and overall momentum.
        """
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # FIXED: Explicitly added the 'tools' parameter for Anthropic
                response = self.client.messages.create(
                    model="claude-3-7-sonnet-20250219", # Ensure this matches your target model
                    max_tokens=1024,
                    tools=[{
                        "name": "web_search",
                        "description": "Search the web for live financial data, DXY, and Treasury yields.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                }
                            },
                            "required": ["query"]
                        }
                    }],
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # ... Keep your existing logic here to parse response.content 
                # and return "BULLISH", "BEARISH", or "BOTH" ...
                
            except Exception as e:
                wait = RETRY_DELAY * attempt
                print(f"🚨 Unexpected error (attempt {attempt}): {e}. Waiting {wait}s...")
                time.sleep(wait)

        print(f"🚨 All {MAX_RETRIES} attempts failed. Defaulting to BOTH.")
        return "BOTH"

    def update_state_file(self, regime: str):
        """Atomically writes the new regime to JSON for the Orchestrator to read."""
        state = {
            "regime_bias": regime,
            "last_updated": datetime.now(UTC).isoformat(),
            "symbols": ["EURUSD","GBPUSD","USDJPY","USDCAD","AUDUSD","USDCHF","EURGBP","EURCHF","XAUUSD","XAGUSD"],
            "grounded": True,
        }
        temp_file = f"{STATE_FILE}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=4)
        os.replace(temp_file, STATE_FILE)
        print(f"✅ Macro Regime updated to: {regime}")

    def run_loop(self):
        print("🤖 Macro Regime Agent initialized. Starting loop with live web search...")
        while True:
            new_regime = self.analyze_market_regime()
            self.update_state_file(new_regime)
            self.current_regime = new_regime
            time.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    agent = MacroRegimeAgent()
    try:
        agent.run_loop()
    except KeyboardInterrupt:
        print("\n[Macro Agent] Shutting down.")