# Description
Algorithmic stock trader built using Pytorch & Alpaca Markets API.

An LSTM model trained to forecast a stock's future prices (5min, 10min, 30min, 60min) given the last 60 minutes of historical data, as well as a confidence measure of its forecast. Additional indicates such as volatility and momentum are also taken into consideration.

The trader applies a momentum strategy to make small winning trades within a 5-60 minute window. Position sizing, risk management, and signal thresholds are all taken into consideration and configurable.

The Alpaca Markets API wrapper utilizes websockets for with retries, REST fallback, and graceful degradation. Market updates are received every 30 seconds, and the API key being used is rate limited to 200 API calls/min, which are key constraints to the system's design.

# How to run
```bash
git clone https://github.com/minghanminghan/stock-trader
pip install requirements.txt
python -m src.model.lstm.training
python -m src.main
```

# TODO
- add unit tests
- retrain LSTM model
- config: split ‘symbols’ into ‘train_symbols’, ‘test_symbols’
- add uuid support, initialize random seeding from config
- create benchmarking tools
- integrate LGBM model
- add more models (XGBoost, NN)
- implement retraining + redeployment
  - redesign
  - training
  - evaluation
  - benchmarking
  - paper trading
  - live trading