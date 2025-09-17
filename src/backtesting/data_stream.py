#!/usr/bin/env python3
"""
Backtesting Data Stream

Replays historical minute-level data to simulate live trading conditions.
Mimics the LiveDataStream interface for seamless integration with existing trading pipeline.

Features:
- Time-based historical data replay
- Configurable speed (real-time, accelerated, or instant)
- Market hours simulation
- Data buffering matching live stream behavior
- Subscriber notification system
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta, timezone
from collections import deque
import threading
import time
from enum import Enum
from dataclasses import dataclass

from src.models._utils.data_ingestion import fetch_stock_data
from src.models._utils.feature_engineering import compute_features
from src.config import DATA_DIR
from src.utils.logging_config import logger


class BacktestMode(Enum):
    """Backtesting operational modes."""
    REALTIME = "realtime"          # 1:1 time simulation
    ACCELERATED = "accelerated"    # Faster than real-time
    INSTANT = "instant"            # Process all data instantly


@dataclass
class BacktestConfig:
    """Configuration for backtesting data stream."""
    start_date: str                    # "YYYY-MM-DD"
    end_date: str                      # "YYYY-MM-DD"
    symbols: List[str]
    mode: BacktestMode = BacktestMode.INSTANT  # Default to instant processing
    market_hours_only: bool = True     # Only replay during market hours
    buffer_size: int = 120             # Minutes of data to keep in buffer
    add_features: bool = True          # Whether to compute technical features


class BacktestDataStream:
    """
    Historical data stream that replays minute-level OHLCV data.

    Provides the same interface as LiveDataStream but uses historical data.
    Supports different replay speeds and market hour simulation.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting data stream.

        Args:
            config: BacktestConfig with backtesting parameters
        """
        self.config = config
        self.is_running = False

        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.data_buffers: Dict[str, deque] = {}
        self.latest_bars: Dict[str, Dict] = {}
        self.subscribers: List[Callable] = []

        # Time tracking
        self.current_time: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.data_index: Dict[str, int] = {}  # Current position in each symbol's data

        # Threading
        self.replay_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Statistics
        self.bars_processed = 0
        self.simulation_start_time: Optional[datetime] = None

        logger.info(f"BacktestDataStream initialized: {config.start_date} to {config.end_date}")
        logger.info(f"Mode: {config.mode.value} (instant processing)")

    def load_historical_data(self) -> bool:
        """
        Load historical data for all symbols in the date range.

        Returns:
            True if data loaded successfully
        """
        logger.info("Loading historical data for backtesting...")

        try:
            # Ensure data is available
            fetch_stock_data(
                tickers=self.config.symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )

            # Load data for each symbol
            for symbol in self.config.symbols:
                file_path = os.path.join(
                    DATA_DIR,
                    f"{symbol}_1min_{self.config.start_date}_to_{self.config.end_date}.parquet"
                )

                if not os.path.exists(file_path):
                    logger.warning(f"No data file found for {symbol}: {file_path}")
                    continue

                # Load symbol data
                df = pd.read_parquet(file_path)

                if df.empty:
                    logger.warning(f"Empty data for {symbol}")
                    continue

                # Reset index and ensure proper format
                df = df.reset_index()
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Filter market hours if requested
                if self.config.market_hours_only:
                    df = self._filter_market_hours(df)

                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)

                # Add features if requested
                if self.config.add_features:
                    try:
                        # Create MultiIndex for feature computation
                        df_indexed = df.set_index(['symbol', 'timestamp'])
                        df_indexed = compute_features(df_indexed)
                        df = df_indexed.reset_index()
                    except Exception as e:
                        logger.warning(f"Feature computation failed for {symbol}: {e}")

                self.historical_data[symbol] = df
                self.data_index[symbol] = 0

                logger.info(f"Loaded {len(df)} bars for {symbol}")

            if not self.historical_data:
                logger.error("No historical data loaded")
                return False

            # Set time boundaries
            all_timestamps = []
            for df in self.historical_data.values():
                if not df.empty:
                    all_timestamps.extend(df['timestamp'].tolist())

            if all_timestamps:
                self.start_time = min(all_timestamps)
                self.end_time = max(all_timestamps)
                self.current_time = self.start_time

                logger.info(f"Data range: {self.start_time} to {self.end_time}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False

    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to only include market hours (9:30 AM - 4:00 PM ET).

        Args:
            df: DataFrame with timestamp column

        Returns:
            Filtered DataFrame
        """
        try:
            # Convert to ET timezone if needed
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter to market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
            market_hours = (
                (df['timestamp'].dt.hour >= 9) &
                (df['timestamp'].dt.hour < 16) &
                (~((df['timestamp'].dt.hour == 9) & (df['timestamp'].dt.minute < 30))) &
                (df['timestamp'].dt.weekday < 5)  # Monday=0, Friday=4
            )

            return df[market_hours].reset_index(drop=True)

        except Exception as e:
            logger.warning(f"Error filtering market hours: {e}")
            return df

    def add_subscriber(self, callback: Callable[[str, Dict], None]) -> None:
        """Add callback for new data notifications."""
        self.subscribers.append(callback)
        logger.debug(f"Added backtesting data subscriber: {callback.__name__}")

    def start_stream(self, symbols: Optional[List[str]] = None) -> None:
        """
        Start the backtesting data stream.

        Args:
            symbols: Optional list to override config symbols
        """
        if self.is_running:
            logger.warning("Backtesting data stream already running")
            return

        if symbols:
            self.config.symbols = symbols

        # Load historical data
        if not self.load_historical_data():
            logger.error("Failed to load historical data")
            return

        # Initialize buffers
        for symbol in self.config.symbols:
            self.data_buffers[symbol] = deque(maxlen=self.config.buffer_size)
            self.latest_bars[symbol] = {}

        self.is_running = True
        self.stop_event.clear()
        self.simulation_start_time = datetime.now()

        # Start replay thread
        if self.config.mode == BacktestMode.INSTANT:
            # Process all data instantly
            self._replay_instant()
        else:
            # Start time-based replay
            self.replay_thread = threading.Thread(
                target=self._replay_loop,
                name="BacktestReplay",
                daemon=True
            )
            self.replay_thread.start()

        logger.info(f"Started backtesting data stream for {len(self.config.symbols)} symbols")

    def stop_stream(self) -> None:
        """Stop the backtesting data stream."""
        if not self.is_running:
            return

        logger.info("Stopping backtesting data stream...")
        self.is_running = False
        self.stop_event.set()

        if self.replay_thread and self.replay_thread.is_alive():
            self.replay_thread.join(timeout=5)

        # Log final statistics
        if self.simulation_start_time:
            elapsed = datetime.now() - self.simulation_start_time
            logger.info(f"Backtesting completed: {self.bars_processed} bars processed in {elapsed}")

        logger.info("Backtesting data stream stopped")

    def _replay_instant(self) -> None:
        """Process all historical data instantly."""
        logger.info("Processing historical data instantly...")

        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for df in self.historical_data.values():
            all_timestamps.update(df['timestamp'].tolist())

        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)

        for timestamp in sorted_timestamps:
            if not self.is_running:
                break

            self.current_time = timestamp
            self._process_timestamp(timestamp)

        logger.info("Instant replay completed")

    def step_forward(self) -> bool:
        """
        Process next timestamp in sequence for event-driven backtesting.

        Returns:
            True if there's more data to process, False if at end
        """
        if not self.historical_data or not self.is_running:
            return False

        # Get all unique timestamps if not already done
        if not hasattr(self, '_sorted_timestamps'):
            all_timestamps = set()
            for df in self.historical_data.values():
                all_timestamps.update(df['timestamp'].tolist())
            self._sorted_timestamps = sorted(all_timestamps)
            self._current_timestamp_index = 0

        # Check if we've reached the end
        if self._current_timestamp_index >= len(self._sorted_timestamps):
            return False

        # Process current timestamp
        current_timestamp = self._sorted_timestamps[self._current_timestamp_index]
        self.current_time = current_timestamp
        self._process_timestamp(current_timestamp)

        # Move to next timestamp
        self._current_timestamp_index += 1

        return self._current_timestamp_index < len(self._sorted_timestamps)

    def reset_to_beginning(self) -> None:
        """Reset the stream to the beginning for a new backtest run."""
        if hasattr(self, '_current_timestamp_index'):
            self._current_timestamp_index = 0
        if hasattr(self, '_sorted_timestamps') and self._sorted_timestamps:
            self.current_time = self._sorted_timestamps[0]

        # Clear buffers
        for symbol in self.data_buffers:
            self.data_buffers[symbol].clear()
            self.latest_bars[symbol] = {}

    def _replay_loop(self) -> None:
        """Main replay loop for time-based simulation."""
        logger.info("Starting time-based replay...")

        # Get all unique timestamps
        all_timestamps = set()
        for df in self.historical_data.values():
            all_timestamps.update(df['timestamp'].tolist())

        sorted_timestamps = sorted(all_timestamps)

        last_real_time = time.time()
        last_sim_time = self.start_time

        for timestamp in sorted_timestamps:
            if not self.is_running or self.stop_event.is_set():
                break

            self.current_time = timestamp

            # Calculate timing for real-time or accelerated replay
            if self.config.mode != BacktestMode.INSTANT:
                sim_elapsed = (timestamp - last_sim_time).total_seconds()

                if self.config.mode == BacktestMode.REALTIME:
                    target_elapsed = sim_elapsed
                else:  # ACCELERATED
                    target_elapsed = sim_elapsed / self.config.speed_multiplier

                real_elapsed = time.time() - last_real_time
                sleep_time = target_elapsed - real_elapsed

                if sleep_time > 0:
                    if self.stop_event.wait(sleep_time):
                        break

            # Process data for this timestamp
            self._process_timestamp(timestamp)

            last_real_time = time.time()
            last_sim_time = timestamp

        logger.info("Time-based replay completed")

    def _process_timestamp(self, timestamp: datetime) -> None:
        """Process all symbol data for a given timestamp."""
        for symbol in self.config.symbols:
            if symbol not in self.historical_data:
                continue

            df = self.historical_data[symbol]

            # Find bars for this timestamp
            matching_bars = df[df['timestamp'] == timestamp]

            for _, row in matching_bars.iterrows():
                bar_data = self._format_bar_data(symbol, row)
                self._process_bar_data(bar_data)
                self.bars_processed += 1

    def _format_bar_data(self, symbol: str, row) -> Dict:
        """Format bar data to standard format."""
        bar_data = {
            'symbol': symbol,
            'timestamp': row['timestamp'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }

        # Add technical features if available
        feature_columns = [
            'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
            'vol_15m', 'vol_60m', 'vol_zscore', 'time_sin', 'time_cos'
        ]

        for col in feature_columns:
            if col in row and not pd.isna(row[col]):
                bar_data[col] = float(row[col])

        return bar_data

    def _process_bar_data(self, bar_data: Dict) -> None:
        """Process and store bar data, notify subscribers."""
        symbol = bar_data['symbol']

        if symbol in self.data_buffers:
            # Update buffers
            self.data_buffers[symbol].append(bar_data)
            self.latest_bars[symbol] = bar_data

            # Notify subscribers
            for callback in self.subscribers:
                try:
                    callback(symbol, bar_data)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")

    # Public API methods (matching LiveDataStream interface)
    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """Get the most recent bar for a symbol."""
        return self.latest_bars.get(symbol)

    def get_historical_bars(self, symbol: str, count: Optional[int] = None) -> List[Dict]:
        """Get historical bars from buffer."""
        if symbol not in self.data_buffers:
            return []

        bars = list(self.data_buffers[symbol])

        if count is not None and count < len(bars):
            return bars[-count:]

        return bars

    def get_recent_data(self, symbol: str, minutes: int = 60) -> List[Dict]:
        """Get recent bars for a symbol."""
        return self.get_historical_bars(symbol, count=minutes)

    def is_data_fresh(self, symbol: str, max_age_minutes: int = 5) -> bool:
        """Check if data for symbol is fresh enough."""
        latest_bar = self.get_latest_bar(symbol)
        if not latest_bar or not self.current_time:
            return False

        try:
            latest_time = pd.to_datetime(latest_bar['timestamp'])
            age = (self.current_time - latest_time.to_pydatetime()).total_seconds() / 60
            return age <= max_age_minutes
        except Exception:
            return False

    def get_ohlcv_dataframe(self, symbol: str, count: Optional[int] = None) -> pd.DataFrame:
        """Get OHLCV data as pandas DataFrame."""
        bars = self.get_historical_bars(symbol, count)

        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        columns = ['open', 'high', 'low', 'close', 'volume']
        return df[columns]

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price (close of latest bar)."""
        latest_bar = self.get_latest_bar(symbol)
        return latest_bar['close'] if latest_bar else None

    def get_stream_status(self) -> Dict:
        """Get comprehensive stream status."""
        return {
            'is_running': self.is_running,
            'mode': 'backtesting',
            'current_time': self.current_time,
            'symbols_tracked': len(self.config.symbols),
            'buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.data_buffers.items()},
            'bars_processed': self.bars_processed,
            'config': {
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'speed_multiplier': self.config.speed_multiplier,
                'market_hours_only': self.config.market_hours_only
            }
        }

    def get_current_timestamp(self) -> Optional[datetime]:
        """Get current simulation timestamp."""
        return self.current_time

    def get_progress(self) -> float:
        """Get simulation progress (0.0 to 1.0)."""
        if not self.start_time or not self.end_time or not self.current_time:
            return 0.0

        total_duration = (self.end_time - self.start_time).total_seconds()
        current_duration = (self.current_time - self.start_time).total_seconds()

        return min(1.0, max(0.0, current_duration / total_duration))

    def seek_to_time(self, target_time: datetime) -> bool:
        """
        Seek to a specific time in the simulation (for instant mode).

        Args:
            target_time: Target timestamp to seek to

        Returns:
            True if seek was successful
        """
        if self.config.mode != BacktestMode.INSTANT:
            logger.warning("Seek only available in instant mode")
            return False

        if not self.start_time or not self.end_time:
            return False

        if target_time < self.start_time or target_time > self.end_time:
            logger.warning(f"Target time {target_time} outside data range")
            return False

        self.current_time = target_time

        # Clear buffers and replay up to target time
        for symbol in self.config.symbols:
            self.data_buffers[symbol].clear()
            self.latest_bars[symbol] = {}

        # Process data up to target time
        for symbol in self.config.symbols:
            if symbol not in self.historical_data:
                continue

            df = self.historical_data[symbol]
            relevant_data = df[df['timestamp'] <= target_time]

            for _, row in relevant_data.iterrows():
                bar_data = self._format_bar_data(symbol, row)
                self._process_bar_data(bar_data)

        return True


def create_backtest_stream(start_date: str,
                          end_date: str,
                          symbols: List[str],
                          market_hours_only: bool = True) -> BacktestDataStream:
    """
    Factory function to create a configured backtesting data stream.

    Args:
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        symbols: List of symbols to backtest
        market_hours_only: Whether to only include market hours

    Returns:
        Configured BacktestDataStream instance
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        mode=BacktestMode.INSTANT,
        market_hours_only=market_hours_only
    )

    return BacktestDataStream(config)


if __name__ == "__main__":
    """Test the backtesting data stream."""

    # Test configuration
    test_config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-02",
        symbols=["AAPL", "MSFT"],
        mode=BacktestMode.INSTANT,
        market_hours_only=True
    )

    # Test callback
    def test_callback(symbol: str, bar_data: Dict):
        print(f"New bar: {symbol} @ {bar_data['timestamp']} - ${bar_data['close']:.2f}")

    # Create and test stream
    stream = BacktestDataStream(test_config)
    stream.add_subscriber(test_callback)

    print("Starting backtest stream test...")
    stream.start_stream()

    # Let it run for a bit
    time.sleep(5)

    # Check status
    status = stream.get_stream_status()
    print(f"Status: {status}")

    # Stop stream
    stream.stop_stream()
    print("Test completed")