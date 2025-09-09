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
- train RNN model on S&P 100 across 3-4 years
  - new output:
    - period, value, confidence
    - model takes a forecasting role
    - rule-based decision-making for actual trading decisions
  - targets metrics to optimize:
    - Sharpe Ratio (risk-adjusted returns)
    - Maximum Drawdown (downside protection)
    - Win Rate (consistency)
    - Profit Factor (wins vs losses ratio)
- refactor codebase to reflect rest of the code 
- include multiple prediction models (XGBoost, NN)
- implement periodic retraining + deployment