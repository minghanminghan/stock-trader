import os
from dotenv import load_dotenv

# Load `environment` variables from .env file
load_dotenv()


# --- Alpaca API Configuration ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or ''
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or ''

if ALPACA_API_KEY == '' or ALPACA_SECRET_KEY == '':
    raise ValueError("Alpaca API keys must be set in the .env file.")


# --- File Paths ---
DATA_DIR = "data"
LOGS_DIR = "logs"


# --- Random Seed ---
RANDOM_SEED = 0

# --- Data Configuration ---
TICKERS = [
    "AAPL", "ADBE", "AMD", "AMZN", "AVGO", "CRM", "CSCO",
    "GOOG", "GOOGL", "IBM", "INTC", "INTU", "META", "MSFT",
    "NFLX", "NOW", "NVDA", "ORCL", "PLTR", "PYPL", "QCOM", "TSLA"
]
TRAINING_START_DATE = "2025-01-01"  # model training
TRAINING_END_DATE = "2025-08-31"

VALIDATE_START_DATE = "2025-09-01"      # model eval
VALIDATE_END_DATE = "2025-09-15"


# --- Backtesting Configuration ---
BACKTESTING_TICKERS = [
    "AAPL", "ADBE", "AMD", "AMZN", "AVGO", "CRM", "CSCO",
    "GOOG", "GOOGL", "IBM", "INTC", "INTU", "META", "MSFT",
    "NFLX", "NOW", "NVDA", "ORCL", "PLTR", "PYPL", "QCOM", "TSLA"
]
BACKTESTING_START_DATE = "2025-09-01"
BACKTESTING_END_DATE = "2025-09-15"

# --- LSTM Model Configuration ---
LSTM_CONFIG = {
    'model': {
        'input_size': 14,
        'sequence_length': 60,
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.2,
        'prediction_horizon': 60  # 60-minute prediction
    },
    'training': {
        'epochs': 150,
        'batch_size': None,  # Will be set dynamically based on available memory
        'early_stopping_patience': 15,  # Stop training if no improvement for 15 epochs
    },
    'optimizer': {
        'type': 'adamw',
        'lr': 0.0005,
        'weight_decay': 1e-5
    },
    'scheduler': {
        'type': 'plateau',
        'patience': 8,
        'factor': 0.5
    }
}

STRATEGY_CONFIG = {
    'buy_threshold': 0.5,         # confidence threshold to buy
    'sell_threshold': 0.5,        # 1% predicted decrease to sell
    'stop_loss_pct': 0.05,        # 5% stop loss
    'take_profit_pct': 0.10,      # 10% take profit
    'max_daily_trades': 50,         # Maximum trades per day
    'max_open_positions': 5,        # Maximum simultaneous positions
    'daily_loss_limit': 500.0,    # Daily loss limit in dollars
}