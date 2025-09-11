# MODEL DEPLOYMENT PIPELINE
1. redesign
2. training
3. evaluation
4. benchmarking
5. paper trading
6. live trading

# SYSTEM INFO
constraints:
- alpaca markets rate limits (200 requests/min)
- computer memory limits (~4gb)
- network limits

system workload:
- model inference:
  - #_SYMBOLS predictions/min
- alpaca api
  - 1 batch GET requests/min
  - [0, #_SYMBOLS] POST requests/min

# TODO
- patch main: on cold start, fetch prev 60 minutes of data using an api request
- patch config: split symbols into train_symbols, trade_symbols
- create unit tests for everything!!!
- retrain LSTM model on S&P 100 across 3-4 years
- update tickers in src/config.py
- look at uuid, seeding random stuff
- create benchmarking tools
- redo LGBM
- add prediction models (XGBoost, NN)
- implement periodic retraining + deployment