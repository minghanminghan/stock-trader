import pandas as pd
import numpy as np
import os
import hashlib
from src.utils.logging_config import logger

def _get_cache_key(df):
    """Generate cache key based on DataFrame content and structure."""
    key_components = [
        str(len(df)),
        str(df.index.min()),
        str(df.index.max()),
        str(df.columns.tolist())
    ]
    key_string = "_".join(key_components)
    return hashlib.md5(key_string.encode()).hexdigest()[:12]

def compute_features(df, use_cache=False):
    """
    Computes specific features for the LSTM model.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
                           Must have a multi-index with ('symbol', 'timestamp').
        use_cache (bool): Whether to use cached features if available.

    Returns:
        pd.DataFrame: The original DataFrame with added feature columns:
                     ['open', 'high', 'low', 'close', 'volume', 'return_1m',
                      'mom_5m', 'mom_15m', 'mom_60m', 'vol_15m', 'vol_60m',
                      'vol_zscore', 'time_sin', 'time_cos']
    """
    # if use_cache:
    #     os.makedirs('data/cache/features', exist_ok=True)
    #     cache_key = _get_cache_key(df)
    #     cache_path = f'data/cache/features/features_{cache_key}.parquet'

    #     if os.path.exists(cache_path):
    #         # logger.info(f"Loading cached features from {cache_path}")
    #         return pd.read_parquet(cache_path)

    # logger.info("Computing features...")

    # Ensure the dataframe is sorted by timestamp for rolling calculations
    df = df.sort_index(level='timestamp')

    # Vectorized feature computation
    df_grouped = df.groupby(level='symbol')

    # 1. Returns (1-minute log returns)
    df['return_1m'] = df_grouped['close'].transform(lambda x: np.log(x / x.shift(1)))

    # 2. Momentum indicators (percentage changes) - computed efficiently
    for period in [5, 15, 60]:
        df[f'mom_{period}m'] = df_grouped['close'].transform(lambda x: x.pct_change(period))

    # 3. Volatility indicators (rolling standard deviation of 1-minute returns)
    df['vol_15m'] = df_grouped['return_1m'].transform(lambda x: x.rolling(15).std())
    df['vol_60m'] = df_grouped['return_1m'].transform(lambda x: x.rolling(60).std())

    # 4. Volume Z-Score (how current volume compares to its rolling average)
    rolling_mean = df_grouped['volume'].transform(lambda x: x.rolling(30).mean())
    rolling_std = df_grouped['volume'].transform(lambda x: x.rolling(30).std())
    df['vol_zscore'] = (df['volume'] - rolling_mean) / rolling_std

    # 5. Time-of-day encoding (cyclical features) - vectorized
    timestamps = df.index.get_level_values('timestamp')
    minutes_in_day = 24 * 60
    time_normalized = (timestamps.hour * 60 + timestamps.minute) / minutes_in_day
    df['time_sin'] = np.sin(2 * np.pi * time_normalized)
    df['time_cos'] = np.cos(2 * np.pi * time_normalized)

    result = df.dropna()

    # # Cache the result
    # if use_cache:
    #     result.to_parquet(cache_path)
    #     logger.info(f"Cached features to {cache_path}")

    return result
