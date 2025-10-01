from v2.config import ENVIRONMENT, ALPACA_HISTORICAL
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
import logging
import os
import pandas as pd
import numpy as np
from typing import Callable
from datetime import datetime


# logger
if ENVIRONMENT == 'dev':
    log_level = logging.DEBUG
else:
    log_level = logging.INFO

logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        handlers=[
            # logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger()

def log_params(fn: Callable):
    def wrapper(*args, **kwargs):
        logger.debug(f"{fn.__name__} called with args={args}, kwargs={kwargs}") 
        return fn(*args, **kwargs)
    return wrapper

def log_params_async(fn: Callable):
    async def wrapper(*args, **kwargs):
        logger.debug(f"{fn.__name__} called with args={args}, kwargs={kwargs}") 
        return await fn(*args, **kwargs)
    return wrapper


@log_params
def get_data(symbol: str, start: datetime, end: datetime, feed: DataFeed = DataFeed.IEX, bypass_local=False) -> pd.DataFrame:
    path = f'{symbol}_1_1Day_{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}' # hardcoding some timeframe stuff here
    full_path = os.path.join(os.getcwd(), 'data', path)
    
    try: # try reading from data folder
        if not bypass_local:
            data = pd.read_parquet(full_path)
            logger.debug(f'successfully read {path}')
        else:
            logger.debug(f'bypassed local, fetching from api')
            data = ALPACA_HISTORICAL.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                feed=feed)
            ).df

    except: # get from ALPACA and save
        logger.debug(f'failed to read {path}, fetching from api')
        data = ALPACA_HISTORICAL.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            feed=feed)
        ).df
        data.to_parquet(full_path)
        logger.debug(f'successfully saved data to {path}')

    finally:
        return data


@log_params
def get_quotes(symbol: str, start: datetime, end: datetime, feed: DataFeed=DataFeed.IEX):
    # mirror get_data
    pass


# @log_params
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame: # maybe convert to torch.Tensor
    '''
    return columns (16 total):
        - log(close)
        - log1p(volume)
        - log1p(number of trades)
        - log(close) - log(vwap)
        - dow_sin(day of week)
        - dow_cos(day of week)
        - pos30_sin(day of month)
        - pos30_cos(day of month)
        - return_1d (1-day log return)
        - return_5d (5-day log return)
        - return_20d (20-day log return)
        - volatility_20d (20-day realized volatility)
        - hl_spread (log(high-low spread))
        - volume_ma_ratio (volume / 20-day volume MA)
        - price_volume_corr (20-day price-volume correlation)
        - rsi_14 (14-day RSI [0,100])
        - ma_distance (distance from 20-day SMA)
        - macd_signal (MACD line - signal line)
    '''
    logger.debug('preprocess_data')
    # Get required columns
    df = data[['close', 'volume', 'trade_count', 'vwap', 'high', 'low', 'open']].copy()

    # Features
    df["log_close"] = np.log(df['close'])
    df["log1p_volume"] = np.log1p(df['volume'])
    df["log1p_trades"] = np.log1p(df['trade_count'])
    df["close_vwap_diff"] = np.log(df['close']) - np.log(df['vwap'])
    df["dow_sin"] = np.sin(2 * np.pi * df.index.get_level_values(1).dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.get_level_values(1).dayofweek / 7)
    df["dom_sin"] = np.sin(2 * np.pi * (df.index.get_level_values(1).day - 1) / 31)
    df["dom_cos"] = np.cos(2 * np.pi * (df.index.get_level_values(1).day - 1) / 31)

    # Price momentum
    df["return_1d"] = df["log_close"] - df["log_close"].shift(1)
    df["return_5d"] = df["log_close"] - df["log_close"].shift(5)
    df["return_20d"] = df["log_close"] - df["log_close"].shift(20)

    # Volatility features
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["hl_spread"] = np.log(df['high']) - np.log(df['low'])

    # Volume features
    df["volume_ma_20"] = df['volume'].rolling(20).mean()
    df["volume_ma_ratio"] = df['volume'] / df["volume_ma_20"].replace(0, np.nan)
    df["price_volume_corr"] = df["return_1d"].rolling(20).corr(df['volume'].pct_change())

    # Technical indicators
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Moving average distance
    df["sma_20"] = df['close'].rolling(20).mean()
    df["ma_distance"] = (df['close'] - df["sma_20"]) / df["sma_20"].replace(0, np.nan)

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9).mean()
    df["macd_signal"] = macd_line - macd_signal

    # Select final feature columns (18 total)
    feature_cols = [
        "log_close", "log1p_volume", "log1p_trades", "close_vwap_diff",
        "dow_sin", "dow_cos", "dom_sin", "dom_cos",
        "return_1d", "return_5d", "return_20d", "volatility_20d", "hl_spread",
        "volume_ma_ratio", "price_volume_corr", "rsi_14", "ma_distance", "macd_signal"
    ]

    result = df[feature_cols].copy()

    # Replace infinite values with NaN
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


if __name__ == '__main__':
    data = get_data('AAPL', datetime(2021, 1, 1), datetime(2021, 12, 31))

    processed_data = preprocess_data(data)
    print(processed_data)