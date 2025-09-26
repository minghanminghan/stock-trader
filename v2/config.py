import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, URL
from alpaca_trade_api.stream import Stream
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
    'input_size': 8,            # log(close), log1p(volume), log1p(number of trades), log(close) - log(vwap), dow_sin/cos, dom_sin/cos
    'input_length': 60,         # 60 days of history
    'output_length': 15,        # 15 days of forecast (log return r_t = log(p_t) - log(p_{t-1})
    'hidden_size': 64,
    'num_layers': 2,            # number of LSTM layers
    'dropout': 0.2,
}

LSTM_TRAINING = {
    'epochs': 100,
    'batch_size': 64,           # dependent on gpu memory
    'validation_split': 0.2,
    'optimizer': 'adamw',
    'lr': 0.001,
    'weight_decay': 1e-5,
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

ALPACA_REST = REST(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    api_version='v2',
    base_url=URL('https://paper-api.alpaca.markets') if paper else URL('https://api.alpaca.markets')
)

ALPACA_STREAM = Stream(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    base_url=URL('wss://paper-api.alpaca.markets/stream') if paper else URL('wss://api.alpaca.markets/stream'),
)