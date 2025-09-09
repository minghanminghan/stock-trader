import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Dict

from src.utils.logging_config import logger


def compute_features_single_symbol(symbol_data: tuple) -> tuple:
    """
    Compute features for a single symbol's data.
    Designed for parallel processing.
    
    Args:
        symbol_data: Tuple of (symbol, dataframe)
        
    Returns:
        Tuple of (symbol, featured_dataframe)
    """
    symbol, df = symbol_data
    
    try:
        # Ensure sorted by timestamp
        df = df.sort_index()
        
        # 1. Returns
        df['return_1m'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Momentum 
        df['mom_5m'] = df['close'].pct_change(5)
        df['mom_15m'] = df['close'].pct_change(15)
        df['mom_60m'] = df['close'].pct_change(60)
        
        # 3. Volatility
        df['vol_15m'] = df['return_1m'].rolling(15).std()
        df['vol_60m'] = df['return_1m'].rolling(60).std()
        
        # Z-score of 15min volatility
        vol_mean = df['vol_15m'].rolling(240).mean()
        vol_std = df['vol_15m'].rolling(240).std()
        df['vol_zscore'] = (df['vol_15m'] - vol_mean) / vol_std
        
        # 4. Time-based features
        if hasattr(df.index, 'hour'):
            hours = df.index.hour
            minutes = df.index.minute
        else:
            # Handle MultiIndex case
            timestamps = df.index.get_level_values('timestamp')
            hours = timestamps.hour
            minutes = timestamps.minute
        
        time_of_day = hours + minutes / 60.0
        df['time_sin'] = np.sin(2 * np.pi * time_of_day / 24)
        df['time_cos'] = np.cos(2 * np.pi * time_of_day / 24)
        
        return (symbol, df)
        
    except Exception as e:
        logger.error(f"Error computing features for {symbol}: {e}")
        return (symbol, df)  # Return original df if feature computation fails


def compute_features_parallel(df: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """
    Compute features in parallel across symbols.
    
    Args:
        df: DataFrame with MultiIndex (symbol, timestamp)
        n_workers: Number of parallel workers (defaults to CPU count - 1)
        
    Returns:
        DataFrame with features added
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    logger.info(f"Computing features in parallel using {n_workers} workers...")
    
    # Group by symbol
    symbol_groups = [(symbol, group.droplevel('symbol')) for symbol, group in df.groupby(level='symbol')]
    
    if len(symbol_groups) <= 1:
        # Use sequential processing for single symbol
        from src.feature_engineering import compute_features
        return compute_features(df)
    
    # Process symbols in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(compute_features_single_symbol, symbol_groups)
    
    # Combine results
    featured_dfs = []
    for symbol, featured_df in results:
        # Restore MultiIndex
        featured_df['symbol'] = symbol
        featured_df = featured_df.set_index('symbol', append=True)
        featured_df = featured_df.reorder_levels(['symbol', featured_df.index.names[0]])
        featured_dfs.append(featured_df)
    
    final_df = pd.concat(featured_dfs)
    
    # Add original columns that might have been missed
    original_columns = ['open', 'high', 'low', 'close', 'volume']
    if 'trade_count' in df.columns:
        original_columns.append('trade_count')
    if 'vwap' in df.columns:
        original_columns.append('vwap')
    
    feature_columns = [
        'return_1m', 'mom_5m', 'mom_15m', 'mom_60m', 
        'vol_15m', 'vol_60m', 'vol_zscore', 
        'time_sin', 'time_cos'
    ]
    
    all_columns = original_columns + feature_columns
    available_columns = [col for col in all_columns if col in final_df.columns]
    
    logger.info(f"Parallel feature computation complete. Added columns: {feature_columns}")
    
    return final_df[available_columns]


class StreamingFeatureComputer:
    """
    Optimized feature computer for live streaming data.
    Maintains rolling windows to avoid recomputing features from scratch.
    """
    
    def __init__(self, max_window_size: int = 300):
        """
        Initialize streaming feature computer.
        
        Args:
            max_window_size: Maximum window size to maintain for rolling calculations
        """
        self.max_window_size = max_window_size
        self.symbol_buffers: Dict[str, pd.DataFrame] = {}
        self.feature_cache: Dict[str, Dict] = {}
        
        logger.info(f"StreamingFeatureComputer initialized with window size: {max_window_size}")
    
    def update_and_compute_features(self, symbol: str, new_bar: Dict) -> Dict:
        """
        Update buffer with new bar and compute incremental features.
        
        Args:
            symbol: Stock symbol
            new_bar: New OHLCV bar data
            
        Returns:
            Dictionary of computed features for the latest bar
        """
        # Initialize buffer if needed
        if symbol not in self.symbol_buffers:
            self.symbol_buffers[symbol] = pd.DataFrame()
            self.feature_cache[symbol] = {}
        
        # Add new bar to buffer
        bar_df = pd.DataFrame([new_bar])
        bar_df['timestamp'] = pd.to_datetime(bar_df['timestamp'])
        bar_df.set_index('timestamp', inplace=True)
        
        # Append to existing buffer
        self.symbol_buffers[symbol] = pd.concat([self.symbol_buffers[symbol], bar_df])
        
        # Trim buffer to max size
        if len(self.symbol_buffers[symbol]) > self.max_window_size:
            self.symbol_buffers[symbol] = self.symbol_buffers[symbol].tail(self.max_window_size)
        
        # Compute features incrementally
        buffer = self.symbol_buffers[symbol]
        
        if len(buffer) < 2:
            return {}  # Need at least 2 bars for returns
        
        try:
            # Compute only latest features (incremental approach)
            latest_idx = buffer.index[-1]
            
            features = {}
            
            # Returns (only need previous close)
            if len(buffer) >= 2:
                features['return_1m'] = np.log(buffer['close'].iloc[-1] / buffer['close'].iloc[-2])
            
            # Momentum features
            if len(buffer) >= 6:
                features['mom_5m'] = (buffer['close'].iloc[-1] / buffer['close'].iloc[-6]) - 1
            if len(buffer) >= 16:
                features['mom_15m'] = (buffer['close'].iloc[-1] / buffer['close'].iloc[-16]) - 1
            if len(buffer) >= 61:
                features['mom_60m'] = (buffer['close'].iloc[-1] / buffer['close'].iloc[-61]) - 1
            
            # Volatility (need returns first)
            if 'return_1m' in features and len(buffer) >= 15:
                # Compute rolling returns for volatility
                returns = np.log(buffer['close'] / buffer['close'].shift(1))
                features['vol_15m'] = returns.tail(15).std()
                
                if len(buffer) >= 60:
                    features['vol_60m'] = returns.tail(60).std()
                
                if len(buffer) >= 240:
                    vol_15m_series = returns.rolling(15).std()
                    vol_mean = vol_15m_series.tail(240).mean()
                    vol_std = vol_15m_series.tail(240).std()
                    if vol_std > 0:
                        features['vol_zscore'] = (features['vol_15m'] - vol_mean) / vol_std
            
            # Time features
            timestamp = latest_idx
            hour = timestamp.hour
            minute = timestamp.minute
            time_of_day = hour + minute / 60.0
            features['time_sin'] = np.sin(2 * np.pi * time_of_day / 24)
            features['time_cos'] = np.cos(2 * np.pi * time_of_day / 24)
            
            # Cache features
            self.feature_cache[symbol] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing streaming features for {symbol}: {e}")
            return self.feature_cache.get(symbol, {})
    
    def get_cached_features(self, symbol: str) -> Dict:
        """Get last computed features for symbol."""
        return self.feature_cache.get(symbol, {})