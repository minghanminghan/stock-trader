import pandas as pd
import numpy as np
from src.utils.logging_config import logger

def compute_features(df):
    """
    Computes a set of features for the given stock data.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data. 
                           Must have a multi-index with ('symbol', 'timestamp').

    Returns:
        pd.DataFrame: The original DataFrame with added feature columns.
    """
    logger.info("Computing features...")
    
    # Ensure the dataframe is sorted by timestamp for rolling calculations
    df = df.sort_index(level='timestamp')

    # 1. Returns
    df['return_1m'] = df.groupby(level='symbol')['close'].transform(lambda x: np.log(x / x.shift(1)))

    # 2. Momentum (based on closing price changes)
    df['mom_5m'] = df.groupby(level='symbol')['close'].transform(lambda x: x.pct_change(5))
    df['mom_15m'] = df.groupby(level='symbol')['close'].transform(lambda x: x.pct_change(15))
    df['mom_60m'] = df.groupby(level='symbol')['close'].transform(lambda x: x.pct_change(60))

    # 3. Volatility (rolling standard deviation of 1-minute returns)
    df['vol_15m'] = df.groupby(level='symbol')['return_1m'].transform(lambda x: x.rolling(15).std())
    df['vol_60m'] = df.groupby(level='symbol')['return_1m'].transform(lambda x: x.rolling(60).std())

    # 4. Volume Indicators
    # Volume Z-Score (how the current volume compares to its rolling average)
    rolling_mean = df.groupby(level='symbol')['volume'].transform(lambda x: x.rolling(30).mean())
    rolling_std = df.groupby(level='symbol')['volume'].transform(lambda x: x.rolling(30).std())
    df['vol_zscore'] = (df['volume'] - rolling_mean) / rolling_std

    # 5. Time-of-day encoding (cyclical features)
    timestamps = df.index.get_level_values('timestamp')
    minutes_in_day = 24 * 60
    df['time_sin'] = np.sin(2 * np.pi * (timestamps.hour * 60 + timestamps.minute) / minutes_in_day)
    df['time_cos'] = np.cos(2 * np.pi * (timestamps.hour * 60 + timestamps.minute) / minutes_in_day)

    logger.info(f"Feature computation complete. Added columns: {df.columns.tolist()}")
    
    return df.dropna()
