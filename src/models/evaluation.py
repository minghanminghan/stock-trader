import numpy as np
import pandas as pd
from src.utils.logging_config import logger

def evaluate_strategy(X, y_pred, fwd_returns):
    """
    Evaluates the trading strategy based on model predictions.

    Args:
        X (pd.DataFrame): DataFrame of features, used for indexing.
        y_pred (np.ndarray): The model's predictions (1, -1, or 0).
        fwd_returns (pd.Series): The forward returns for the corresponding period.

    Returns:
        dict: A dictionary containing key performance metrics.
    """
    logger.info("Evaluating trading strategy performance...")

    if not isinstance(X.index, pd.MultiIndex):
        raise ValueError("Input DataFrame must have a multi-index.")

    # Align predictions with the forward returns
    strategy_df = pd.DataFrame({
        'prediction': y_pred,
        'fwd_return': fwd_returns
    }, index=X.index)

    # Calculate the strategy returns (trade only on non-zero signals)
    strategy_df['strategy_return'] = strategy_df['prediction'] * strategy_df['fwd_return']

    # --- Calculate Metrics ---
    # 1. Net P&L per trade (assuming a fixed cost per trade)
    transaction_cost = 0.0005  # 0.05% per trade
    trades = strategy_df[strategy_df['prediction'] != 0]
    net_pnl_per_trade = (trades['strategy_return'] - transaction_cost).mean()

    # 2. Sharpe Ratio (annualized)
    # Assuming daily returns for annualization. This is a simplification.
    # A more accurate approach would require resampling to daily.
    timestamps = pd.to_datetime(strategy_df.index.get_level_values('timestamp'))
    daily_returns = strategy_df.groupby(timestamps.date)['strategy_return'].sum()
    if daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) # Annualized

    # 3. Hit Rate (fraction of profitable trades)
    profitable_trades = (trades['strategy_return'] > 0).sum()
    total_trades = len(trades)
    hit_rate = profitable_trades / total_trades if total_trades > 0 else 0.0

    metrics = {
        'net_pnl_per_trade': net_pnl_per_trade,
        'sharpe_ratio': sharpe_ratio,
        'hit_rate': hit_rate,
        'total_trades': total_trades
    }

    logger.info("Strategy Evaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"  - {key}: {value:.4f}")

    return metrics
