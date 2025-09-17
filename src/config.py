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


# --- LSTM Model Configuration ---
LSTM_CONFIG = {
    'model': {
        'input_size': 14,
        'sequence_length': 60,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'prediction_horizon': 60  # 60-minute prediction
    },
    'training': {
        'epochs': 100,
        'batch_size': None,  # Will be set dynamically based on available memory
        'early_stopping_patience': 15,  # Reduced patience
        'dataset_size_fraction': 1.0,  # Use 100% of data for training (1.0 = full dataset)
    },
    'optimizer': {
        'type': 'adamw',
        'lr': 0.001,
        'weight_decay': 1e-6  # Reduced weight decay
    },
    'scheduler': {
        'type': 'plateau',
        'patience': 8,
        'factor': 0.5
    }
}
