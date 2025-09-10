import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Alpaca API Configuration ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or ''
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or ''

if ALPACA_API_KEY == '' or ALPACA_SECRET_KEY == '':
    raise ValueError("Alpaca API keys must be set in the .env file.")

# --- Data Configuration ---
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA"]
TRAINING_START_DATE = "2022-01-01"
TRAINING_END_DATE = "2023-01-01"
TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2023-06-01"
START_DATE = "2022-01-01"
END_DATE = "2024-01-01"

# --- File Paths ---
DATA_DIR = "data"
LOGS_DIR = "logs"
