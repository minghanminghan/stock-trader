import os
import pandas as pd
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, DATA_DIR
from src.utils.logging_config import logger

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches historical minute-level OHLCV data for a list of tickers from Alpaca.
    Implements caching to avoid re-downloading existing data.

    Args:
        tickers (list): A list of stock tickers to fetch.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
    """
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    for ticker in tickers:
        logger.info(f"Processing data for {ticker}...")
        
        # Caching logic: check if the file already exists
        file_path = os.path.join(DATA_DIR, f"{ticker}_1min_{start_date}_to_{end_date}.parquet")
        if os.path.exists(file_path):
            logger.info(f"Data for {ticker} already exists locally. Skipping download.")
            continue

        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed="sip"
            )

            bars = client.get_stock_bars(request_params).df

            if bars.empty:
                logger.warning(f"No data returned for {ticker} for the given date range.")
                continue

            # Localize timestamps to US market time
            bars = bars.reset_index()
            bars['timestamp'] = pd.to_datetime(bars['timestamp']) #.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            bars = bars.set_index(['symbol', 'timestamp'])

            # Ensure the data directory exists
            os.makedirs(DATA_DIR, exist_ok=True)
            
            bars.to_parquet(file_path)
            logger.info(f"Successfully downloaded and saved data for {ticker} to {file_path}")

        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")

if __name__ == '__main__':
    # Example usage:
    # from src.config import TICKERS, TRAINING_START_DATE, TRAINING_END_DATE
    # fetch_stock_data(tickers=TICKERS, start_date=TRAINING_START_DATE, end_date=TRAINING_END_DATE)
    fetch_stock_data(tickers=["AAPL", "MSFT"], start_date="2023-01-01", end_date="2023-01-02")