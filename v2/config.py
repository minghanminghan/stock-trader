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
    'investment_size': 0.02,        # 2% of cash per trade
    'buy_threshold_log': 0.2,       # predicted log increase to buy
    'sell_threshold_log': 0.2,      # predicted log decrease to sell
    'take_profit': 1.2,             # 20% take profit
    'stop_loss': 0.8,               # 20% stop loss
}
SYMBOLS = ['AAPL']


# model
LSTM_MODEL = {
    'input_size': 16,           # feature set
    'input_length': 60,         # 60 days of history
    'output_length': 15,        # 15 days of forecast (log return r_t = log(p_t) - log(p_{t-1})
    'hidden_size': 64,          # 64 hidden features
    'num_layers': 2,            # number of LSTM layers
    'dropout': 0.2,
}


LSTM_TRAINING = {
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