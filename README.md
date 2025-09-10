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
- train LSTM model on S&P 100 across 3-4 years
  - update tickers in src/config.py
  - new output:
    - period, value, confidence
    - model takes a forecasting role
    - rule-based decision-making for actual trading decisions
  - targets metrics to optimize:
    - Sharpe Ratio (risk-adjusted returns)
    - Maximum Drawdown (downside protection)
    - Win Rate (consistency)
    - Profit Factor (wins vs losses ratio)
- look at uuid, seeding random stuff
- create benchmarking tools
- create unit tests for alpaca/, models/, trading/
- redo LGBM
- add prediction models (XGBoost, NN)
- implement periodic retraining + deployment