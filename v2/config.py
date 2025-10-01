from datetime import datetime
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
load_dotenv()


# environment
ENVIRONMENT = 'dev' # dev, prod
RANDOM_SEED = 0


# strategy
STRATEGY = {
    'buying_power': 0,              # cash + stock value
    'cash': 0,                      # cash on hand
    'investment_size': 0.05,        # 2% of cash per trade
    'buy_threshold_log': 0.02,       # predicted log increase to buy (2% gain)
    'sell_threshold_log': -0.02,     # predicted log decrease to sell (2% loss)
    'take_profit': 1.2,             # 20% take profit
    'stop_loss': 0.8,               # 20% stop loss
}
SYMBOLS = ['AAPL']


# model
LSTM_MODEL = {
    'input_size': 18,           # feature set
    'input_length': 60,         # 60 days of history
    'output_size': 18,          # match input size for iterative forecasting
    'output_length': 15,        # 15 days of forecast (log return r_t = log(p_t) - log(p_{t-1})
    'hidden_size': 64,          # 64 hidden features
    'num_layers': 2,            # number of LSTM layers
    'dropout': 0.2,
}


LSTM_TRAINING = {
    'train_symbols': [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
        'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK.B', 'C',
        'CAT', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX',
        'DE', 'DHR', 'DIS', 'DUK', 'EMR', 'FDX', 'GD', 'GE', 'GILD', 'GM',
        'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'INTU', 'ISRG', 'JNJ',
        'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT',
        'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE',
        'NOW', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PLTR', 'PM', 'PYPL', 'QCOM',
        'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 'TSLA',
        'TXN', 'UBER', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
    ],
    'epochs': 100,
    'batch_size': 64,               # dependent on gpu memory
    'optimizer': 'adamw',
    'lr': 0.001,
    'weight_decay': 1e-4,
    'mse_weight': 0.7,
    'mae_weight': 0.3,
    'lr_scheduler': True,
    'lr_patience': 10,
    'lr_factor': 0.5,
    'early_stopping_patience': 15,
    'gradient_clip_norm': 1.0,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'start_date': datetime(2005, 1, 1),
    'end_date': datetime(2025, 9, 1),
}


# backtest
BACKTEST = {
    'symbols': [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV',
        # Consumer/Industrial
        'KO', 'PG', 'WMT', 'CAT', 'BA',
        # Energy/Materials
        'XOM', 'CVX', 'LIN',
        # Different sectors for robustness testing
    ],
    'start_date': datetime(2020, 1, 1),
    'end_date': datetime(2023, 12, 31),
    'initial_capital': 100000,
    'commission': 0.005,    # per trade
    'slippage': 0.001,      # price impact
}


# alpaca api
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or ''
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or ''

if ALPACA_API_KEY == '' or ALPACA_SECRET_KEY == '':
    raise ValueError("Alpaca API keys must be set in the .env file.")
if ENVIRONMENT == 'dev':
    paper = True
else:
    paper = False

ALPACA_CLIENT = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=paper
)
ALPACA_STREAM = StockDataStream(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)
ALPACA_HISTORICAL = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)