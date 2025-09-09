from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import asyncio
import threading
import time

from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from src.utils.logging_config import logger


class LiveDataStream:
    """
    Real-time market data feed from Alpaca.
    Streams minute-level OHLCV data and maintains rolling buffers.
    """
    
    def __init__(self, buffer_size: int = 100):
        """
        Initialize live data stream.
        
        Args:
            buffer_size: Number of bars to keep in rolling buffer per symbol
        """
        self.historical_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.stream_client = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        self.buffer_size = buffer_size
        self.data_buffers: Dict[str, deque] = {}  # Rolling buffers per symbol
        self.latest_bars: Dict[str, Dict] = {}    # Most recent bar per symbol
        self.subscribers: List[Callable] = []     # Callback functions
        
        self.is_streaming = False
        self.stream_thread = None
        self.last_heartbeat = datetime.now()
        self.heartbeat_timeout = 300  # 5 minutes
        
        logger.info(f"LiveDataStream initialized with buffer size: {buffer_size}")
    
    def add_subscriber(self, callback: Callable[[str, Dict], None]) -> None:
        """
        Add callback function to be notified of new data.
        
        Args:
            callback: Function to call with (symbol, bar_data)
        """
        self.subscribers.append(callback)
        logger.debug(f"Added data subscriber: {callback.__name__}")
    
    def start_stream(self, symbols: List[str]) -> None:
        """
        Start streaming real-time data for symbols.
        
        Args:
            symbols: List of stock symbols to stream
        """
        if self.is_streaming:
            logger.warning("Data stream already running")
            return
        
        # Initialize buffers
        for symbol in symbols:
            self.data_buffers[symbol] = deque(maxlen=self.buffer_size)
            self.latest_bars[symbol] = {}
        
        # Populate initial historical data
        self._load_initial_data(symbols)
        
        # Start streaming thread
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._run_stream, args=(symbols,))
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        logger.info(f"Started streaming data for {len(symbols)} symbols")
    
    def stop_stream(self) -> None:
        """Stop the data stream."""
        self.is_streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        logger.info("Data stream stopped")
    
    def _load_initial_data(self, symbols: List[str]) -> None:
        """Load initial historical data to populate buffers."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=5)  # 5 days of history
        
        for symbol in symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Minute,
                    start=start_time,
                    end=end_time
                )
                
                bars = self.historical_client.get_stock_bars(request).df
                
                if not bars.empty:
                    bars = bars.reset_index()
                    
                    # Process each bar
                    for _, row in bars.iterrows():
                        bar_data = {
                            'symbol': symbol,
                            'timestamp': row['timestamp'],
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': int(row['volume'])
                        }
                        
                        self.data_buffers[symbol].append(bar_data)
                        self.latest_bars[symbol] = bar_data
                
                logger.debug(f"Loaded {len(self.data_buffers[symbol])} historical bars for {symbol}")
                time.sleep(0.1)  # Rate limiting
                
            except APIError as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading data for {symbol}: {e}")
    
    def _run_stream(self, symbols: List[str]) -> None:
        """Run the asyncio streaming loop in a separate thread."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Subscribe to bar updates
            self.stream_client.subscribe_bars(self._handle_bar_update, *symbols)
            
            # Start streaming
            logger.info("Starting Alpaca data stream...")
            loop.run_until_complete(self._stream_with_heartbeat())
            
        except Exception as e:
            logger.error(f"Error in streaming thread: {e}")
            self.is_streaming = False
        finally:
            try:
                loop.close()
            except:
                pass
    
    async def _stream_with_heartbeat(self) -> None:
        """Stream data with heartbeat monitoring."""
        try:
            # Start the stream
            stream_task = asyncio.create_task(self.stream_client._run_forever())
            
            # Heartbeat monitoring
            while self.is_streaming:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check heartbeat
                time_since_heartbeat = datetime.now() - self.last_heartbeat
                if time_since_heartbeat.seconds > self.heartbeat_timeout:
                    logger.error("Data stream heartbeat timeout")
                    break
            
            # Cancel stream task
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            logger.error(f"Error in stream heartbeat: {e}")
    
    async def _handle_bar_update(self, bar) -> None:
        """
        Handle new bar data from stream.
        
        Args:
            bar: Bar data from Alpaca stream
        """
        try:
            symbol = bar.symbol
            self.last_heartbeat = datetime.now()
            
            # Convert to standard format
            bar_data = {
                'symbol': symbol,
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            }
            
            # Update buffers
            if symbol in self.data_buffers:
                self.data_buffers[symbol].append(bar_data)
                self.latest_bars[symbol] = bar_data
                
                # Notify subscribers
                for callback in self.subscribers:
                    try:
                        callback(symbol, bar_data)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {e}")
                
                logger.debug(f"New bar: {symbol} @ {bar.timestamp} - Close: ${bar.close:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing bar update: {e}")
    
    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent bar for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest bar data or None
        """
        return self.latest_bars.get(symbol)
    
    def get_historical_bars(self, symbol: str, count: int = None) -> List[Dict]:
        """
        Get historical bars from buffer.
        
        Args:
            symbol: Stock symbol
            count: Number of recent bars (None for all)
            
        Returns:
            List of bar data
        """
        if symbol not in self.data_buffers:
            return []
        
        bars = list(self.data_buffers[symbol])
        
        if count is not None and count < len(bars):
            return bars[-count:]
        
        return bars
    
    def get_recent_data(self, symbol: str, minutes: int = 60) -> List[Dict]:
        """
        Get recent bars for a symbol (alias for get_historical_bars for compatibility).
        
        Args:
            symbol: Stock symbol
            minutes: Number of recent bars to return
            
        Returns:
            List of recent bar data
        """
        return self.get_historical_bars(symbol, count=minutes)
    
    def is_data_fresh(self, symbol: str, max_age_minutes: int = 5) -> bool:
        """
        Check if data for symbol is fresh enough.
        
        Args:
            symbol: Stock symbol
            max_age_minutes: Maximum age in minutes
            
        Returns:
            True if data is fresh, False otherwise
        """
        latest_bar = self.get_latest_bar(symbol)
        if not latest_bar:
            return False
        
        try:
            latest_time = pd.to_datetime(latest_bar['timestamp'])
            age = datetime.now() - latest_time.to_pydatetime()
            return age.total_seconds() / 60 <= max_age_minutes
        except Exception:
            return False
    
    def get_ohlcv_dataframe(self, symbol: str, count: int = None) -> pd.DataFrame:
        """
        Get OHLCV data as pandas DataFrame.
        
        Args:
            symbol: Stock symbol
            count: Number of recent bars
            
        Returns:
            DataFrame with OHLCV data
        """
        bars = self.get_historical_bars(symbol, count)
        
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price (close of latest bar).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None
        """
        latest_bar = self.get_latest_bar(symbol)
        return latest_bar['close'] if latest_bar else None
    
    def is_data_fresh(self, symbol: str, max_age_minutes: int = 5) -> bool:
        """
        Check if data for symbol is fresh (recent).
        
        Args:
            symbol: Stock symbol
            max_age_minutes: Maximum age in minutes
            
        Returns:
            True if data is fresh
        """
        latest_bar = self.get_latest_bar(symbol)
        if not latest_bar:
            return False
        
        bar_time = latest_bar['timestamp']
        if isinstance(bar_time, str):
            bar_time = pd.to_datetime(bar_time)
        
        age = datetime.now(bar_time.tzinfo) - bar_time
        return age.seconds < (max_age_minutes * 60)
    
    def get_stream_status(self) -> Dict:
        """Get current stream status."""
        return {
            'is_streaming': self.is_streaming,
            'symbols_tracked': list(self.data_buffers.keys()),
            'buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.data_buffers.items()},
            'last_heartbeat': self.last_heartbeat,
            'latest_timestamps': {
                symbol: bar.get('timestamp') for symbol, bar in self.latest_bars.items()
            }
        }