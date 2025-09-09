import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Alpaca API Configuration ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Alpaca API keys must be set in the .env file.")

# --- Data Configuration ---
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA"]
START_DATE = "2022-01-01"
END_DATE = "2024-01-01"

# --- File Paths ---
DATA_DIR = "data"
LOGS_DIR = "logs"
