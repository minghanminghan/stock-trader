import os
from dotenv import load_dotenv

# Load environment variables from .env file
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
TICKERS = ["AMZN", "META", "NVDA"] # change to S&P 100

TRAINING_START_DATE = "2025-07-01"  # for model training
TRAINING_END_DATE = "2025-08-31"

VALIDATE_START_DATE = "2025-09-01"      # for model eval
VALIDATE_END_DATE = "2025-09-03"


# --- LSTM Model Configuration ---
LSTM_CONFIG = {
    'model': {
        'input_size': 14,
        'sequence_length': 60,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'prediction_horizons': [5, 15, 30, 60]
    },
    'training': {
        'epochs': 5, # 100
        'batch_size': 64,
        'validation_split': 0.2,
        'early_stopping_patience': 20,
        'gradient_clip_norm': 1.0
    },
    'optimizer': {
        'type': 'adamw',
        'lr': 0.001,
        'weight_decay': 1e-5
    },
    'scheduler': {
        'type': 'plateau',
        'patience': 10,
        'factor': 0.5
    },
    'loss': {
        'horizon_weights': { # skew to short-term gains
            '5min': 2.0,
            '15min': 1.5,
            '30min': 1.0,
            '60min': 0.8
        }
    }
}
