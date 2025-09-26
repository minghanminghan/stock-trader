from v2.config import ENVIRONMENT, ALPACA_REST
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import logging
import os
import pandas as pd
import numpy as np
from typing import Literal, Callable


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
def get_data(symbol: str, start: str, end: str, timeframe_unit: TimeFrameUnit, timeframe_amount: int = 1, feed='iex', bypass_local=False) -> pd.DataFrame:
    logger.debug('get_data')
    path = f'{symbol}_{timeframe_amount}_{timeframe_unit}_{start}_{end}'
    full_path = os.path.join(os.getcwd(), 'data', path)
    
    try: # try reading from data folder
        if not bypass_local:
            data = pd.read_parquet(full_path)
            logger.debug(f'successfully read {path}')
        else:
            logger.debug(f'bypassed local, fetching from api')
            data = ALPACA_REST.get_bars(symbol, TimeFrame(timeframe_amount, timeframe_unit), start, end, feed=feed).df

    except: # get from ALPACA and save
        logger.debug(f'failed to read {path}, fetching from api')
        data = ALPACA_REST.get_bars(symbol, TimeFrame(timeframe_amount, timeframe_unit), start, end, feed=feed).df
        data.to_parquet(full_path)
        logger.debug(f'successfully saved data to {path}')

    finally:
        return data


# @log_params
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame: # maybe convert to torch.Tensor
    '''
    return columns:
        - log(close)
        - log1p(volume)
        - log1p(number of trades)
        - log(close) - log(vwap)
        - dow_sin(day of week)
        - dow_cos(day of week)
        - pos30_sin(day of month)
        - pos30_cos(day of month)
    '''
    logger.debug('preprocess_data')
    # apply to data: log(close), log1p(volume), log1p(number of trades), log(close) - log(vwap), dow_sin/cos, dom_sin/cos
    df = data[['close', 'volume', 'trade_count', 'vwap']]
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["dom_sin"] = np.sin(2 * np.pi * (df.index.day - 1) / 31)
    df["dom_cos"] = np.cos(2 * np.pi * (df.index.day - 1) / 31)

    return df


if __name__ == '__main__':
    data = get_data('AAPL', "2021-06-01", "2021-06-30", TimeFrameUnit.Day)

    processed_data = preprocess_data(data)
    print(processed_data)