#!/usr/bin/env python3
"""
Enhanced Live Data Stream with WebSocket + REST Fallback

Features:
- Primary: WebSocket streaming for real-time data
- Fallback: REST API polling when WebSocket fails
- Retry mechanism with exponential backoff
- Connection health monitoring
- Graceful degradation
"""

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
import random
from enum import Enum

from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from src.utils.logging_config import logger


class DataStreamMode(Enum):
    """Data stream operational modes."""
    WEBSOCKET = "websocket"
    REST_FALLBACK = "rest_fallback"
    OFFLINE = "offline"


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class LiveDataStream:
    """
    Robust live data stream with WebSocket primary + REST fallback.
    Automatically handles connection failures and provides seamless data flow.
    """
    
    def __init__(self, 
                 buffer_size: int = 100,
                 max_retries: int = 5,
                 retry_backoff_base: float = 2.0,
                 fallback_interval: int = 60):
        """
        Initialize enhanced live data stream.
        
        Args:
            buffer_size: Number of bars to keep in rolling buffer per symbol
            max_retries: Maximum connection retry attempts
            retry_backoff_base: Base for exponential backoff (seconds)
            fallback_interval: REST polling interval when WebSocket fails (seconds)
        """
        self.historical_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.stream_client = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        self.buffer_size = buffer_size
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.fallback_interval = fallback_interval
        
        # Data storage
        self.data_buffers: Dict[str, deque] = {}
        self.latest_bars: Dict[str, Dict] = {}
        self.subscribers: List[Callable] = []
        
        # Stream state
        self.symbols: List[str] = []
        self.is_running = False
        self.current_mode = DataStreamMode.OFFLINE
        self.connection_state = ConnectionState.DISCONNECTED
        
        # Threading
        self.stream_thread = None
        self.fallback_thread = None
        self.monitor_thread = None
        
        # Connection monitoring
        self.last_data_received = datetime.now()
        self.heartbeat_timeout = 300  # 5 minutes
        self.retry_count = 0
        self.consecutive_failures = 0
        
        # Performance metrics
        self.websocket_data_count = 0
        self.rest_data_count = 0
        self.connection_attempts = 0
        self.last_connection_time = None
        
        logger.info(f"Enhanced LiveDataStream initialized - buffer: {buffer_size}, max_retries: {max_retries}")
    
    def add_subscriber(self, callback: Callable[[str, Dict], None]) -> None:
        """Add callback for new data notifications."""
        self.subscribers.append(callback)
        logger.debug(f"Added data subscriber: {callback.__name__}")
    
    def start_stream(self, symbols: List[str]) -> None:
        """
        Start streaming data with WebSocket primary + REST fallback.
        
        Args:
            symbols: List of stock symbols to track
        """
        if self.is_running:
            logger.warning("Data stream already running")
            return
        
        self.symbols = symbols
        self.is_running = True
        
        # Initialize buffers
        for symbol in symbols:
            self.data_buffers[symbol] = deque(maxlen=self.buffer_size)
            self.latest_bars[symbol] = {}
        
        # Load initial historical data
        self._load_initial_data(symbols)
        
        # Start primary WebSocket stream
        self._start_websocket_stream()
        
        # Start connection monitor
        self._start_connection_monitor()
        
        logger.info(f"Started enhanced data stream for {len(symbols)} symbols")
    
    def stop_stream(self) -> None:
        """Stop all data streaming."""
        logger.info("Stopping data stream...")
        self.is_running = False
        
        # Stop all threads
        threads = [self.stream_thread, self.fallback_thread, self.monitor_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=3)
        
        self.current_mode = DataStreamMode.OFFLINE
        self.connection_state = ConnectionState.DISCONNECTED
        
        logger.info("Data stream stopped")
    
    def _load_initial_data(self, symbols: List[str]) -> None:
        """Load recent historical data to warm up buffers."""
        logger.info("Loading initial historical data...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)  # 1 day of history
        
        # Batch request for all symbols
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time
            )
            
            bars = self.historical_client.get_stock_bars(request).df
            
            if not bars.empty:
                bars = bars.reset_index()
                
                # Group by symbol and process
                for symbol in symbols:
                    symbol_bars = bars[bars['symbol'] == symbol]
                    
                    if not symbol_bars.empty:
                        for _, row in symbol_bars.iterrows():
                            bar_data = self._format_bar_data(symbol, row)
                            self.data_buffers[symbol].append(bar_data)
                            self.latest_bars[symbol] = bar_data
                        
                        logger.info(f"Loaded {len(symbol_bars)} historical bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading initial historical data: {e}")
            # Continue without initial data
    
    def _start_websocket_stream(self) -> None:
        """Start WebSocket streaming in separate thread."""
        self.stream_thread = threading.Thread(
            target=self._run_websocket_stream,
            name="WebSocketStream",
            daemon=True
        )
        self.stream_thread.start()
    
    def _run_websocket_stream(self) -> None:
        """Run WebSocket stream with retry logic."""
        while self.is_running:
            try:
                self.connection_state = ConnectionState.CONNECTING
                self.connection_attempts += 1
                
                logger.info(f"Attempting WebSocket connection (attempt {self.connection_attempts})")
                
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Subscribe to bar updates
                self.stream_client.subscribe_bars(self._handle_websocket_bar, *self.symbols)
                
                # Start streaming
                self.connection_state = ConnectionState.CONNECTED
                self.current_mode = DataStreamMode.WEBSOCKET
                self.last_connection_time = datetime.now()
                self.retry_count = 0
                self.consecutive_failures = 0
                
                logger.info("WebSocket connected successfully")
                
                # Run until connection fails or stopped
                loop.run_until_complete(self.stream_client._run_forever())
                
            except Exception as e:
                logger.error(f"WebSocket stream error: {e}")
                self.consecutive_failures += 1
                
                if not self.is_running:
                    break
                
                self.connection_state = ConnectionState.RECONNECTING
                
                # Exponential backoff with jitter
                if self.retry_count < self.max_retries:
                    delay = min(
                        self.retry_backoff_base ** self.retry_count + random.uniform(0, 1),
                        60  # Max 60 seconds
                    )
                    logger.info(f"Retrying WebSocket connection in {delay:.1f} seconds...")
                    time.sleep(delay)
                    self.retry_count += 1
                else:
                    logger.error("Max WebSocket retries reached, switching to REST fallback")
                    self.connection_state = ConnectionState.FAILED
                    self._start_rest_fallback()
                    break
            
            finally:
                try:
                    loop.close()
                except:
                    pass
    
    async def _handle_websocket_bar(self, bar) -> None:
        """Handle WebSocket bar data."""
        try:
            symbol = bar.symbol
            self.last_data_received = datetime.now()
            self.websocket_data_count += 1
            
            # Format bar data
            bar_data = {
                'symbol': symbol,
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            }
            
            self._process_bar_data(bar_data)
            
            logger.debug(f"WebSocket bar: {symbol} @ {bar.timestamp} - ${bar.close:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing WebSocket bar: {e}")
    
    def _start_rest_fallback(self) -> None:
        """Start REST API fallback polling."""
        if self.fallback_thread and self.fallback_thread.is_alive():
            return
        
        logger.info("Starting REST API fallback")
        self.current_mode = DataStreamMode.REST_FALLBACK
        
        self.fallback_thread = threading.Thread(
            target=self._run_rest_fallback,
            name="RESTFallback",
            daemon=True
        )
        self.fallback_thread.start()
    
    def _run_rest_fallback(self) -> None:
        """Run REST API polling fallback."""
        last_poll_time = datetime.now() - timedelta(minutes=5)
        
        while self.is_running and self.current_mode == DataStreamMode.REST_FALLBACK:
            try:
                current_time = datetime.now()
                
                # Get latest bars for all symbols
                request = StockBarsRequest(
                    symbol_or_symbols=self.symbols,
                    timeframe=TimeFrame.Minute,
                    start=last_poll_time,
                    end=current_time
                )
                
                bars = self.historical_client.get_stock_bars(request).df
                
                if not bars.empty:
                    bars = bars.reset_index()
                    new_bars_count = 0
                    
                    # Process new bars
                    for _, row in bars.iterrows():
                        symbol = row['symbol']
                        bar_timestamp = row['timestamp']
                        
                        # Only process bars newer than last poll
                        if bar_timestamp > last_poll_time:
                            bar_data = self._format_bar_data(symbol, row)
                            self._process_bar_data(bar_data)
                            new_bars_count += 1
                    
                    if new_bars_count > 0:
                        self.rest_data_count += new_bars_count
                        logger.debug(f"REST fallback: processed {new_bars_count} new bars")
                
                last_poll_time = current_time
                self.last_data_received = current_time
                
                # Sleep until next poll
                time.sleep(self.fallback_interval)
                
            except Exception as e:
                logger.error(f"REST fallback error: {e}")
                time.sleep(30)  # Wait before retry
    
    def _start_connection_monitor(self) -> None:
        """Start connection health monitoring."""
        self.monitor_thread = threading.Thread(
            target=self._run_connection_monitor,
            name="ConnectionMonitor",
            daemon=True
        )
        self.monitor_thread.start()
    
    def _run_connection_monitor(self) -> None:
        """Monitor connection health and trigger fallbacks."""
        while self.is_running:
            try:
                current_time = datetime.now()
                time_since_data = current_time - self.last_data_received
                
                # Check data freshness
                if time_since_data.total_seconds() > self.heartbeat_timeout:
                    logger.warning(f"No data received for {time_since_data.total_seconds():.0f} seconds")
                    
                    # If WebSocket is supposed to be connected but no data
                    if self.current_mode == DataStreamMode.WEBSOCKET:
                        logger.error("WebSocket appears stalled, triggering fallback")
                        self._start_rest_fallback()
                    
                # Try to recover WebSocket if using REST fallback
                elif (self.current_mode == DataStreamMode.REST_FALLBACK and
                      time_since_data.total_seconds() < 120):  # Data is flowing well
                    
                    # Attempt WebSocket recovery every 10 minutes
                    if (self.last_connection_time and 
                        (current_time - self.last_connection_time).total_seconds() > 600):
                        
                        logger.info("Attempting WebSocket recovery...")
                        self.retry_count = 0  # Reset retry count
                        self._start_websocket_stream()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Connection monitor error: {e}")
                time.sleep(30)
    
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
    
    def _format_bar_data(self, symbol: str, row) -> Dict:
        """Format bar data to standard format."""
        return {
            'symbol': symbol,
            'timestamp': row['timestamp'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }
    
    # Public API methods (unchanged from original)
    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """Get the most recent bar for a symbol."""
        return self.latest_bars.get(symbol)
    
    def get_historical_bars(self, symbol: str, count: int|None = None) -> List[Dict]:
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
        if not latest_bar:
            return False
        
        try:
            latest_time = pd.to_datetime(latest_bar['timestamp'])
            age = datetime.now() - latest_time.to_pydatetime()
            return age.total_seconds() / 60 <= max_age_minutes
        except Exception:
            return False
    
    def get_ohlcv_dataframe(self, symbol: str, count: int|None = None) -> pd.DataFrame:
        """Get OHLCV data as pandas DataFrame."""
        bars = self.get_historical_bars(symbol, count)
        
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price (close of latest bar)."""
        latest_bar = self.get_latest_bar(symbol)
        return latest_bar['close'] if latest_bar else None
    
    def get_stream_status(self) -> Dict:
        """Get comprehensive stream status."""
        return {
            'is_running': self.is_running,
            'mode': self.current_mode.value,
            'connection_state': self.connection_state.value,
            'symbols_tracked': len(self.symbols),
            'buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.data_buffers.items()},
            'last_data_received': self.last_data_received,
            'websocket_data_count': self.websocket_data_count,
            'rest_data_count': self.rest_data_count,
            'connection_attempts': self.connection_attempts,
            'consecutive_failures': self.consecutive_failures,
            'retry_count': self.retry_count,
            'uptime': (datetime.now() - self.last_connection_time).total_seconds() if self.last_connection_time else 0
        }
    
    def force_websocket_recovery(self) -> None:
        """Manually trigger WebSocket recovery attempt."""
        if self.current_mode == DataStreamMode.REST_FALLBACK:
            logger.info("Manual WebSocket recovery triggered")
            self.retry_count = 0
            self._start_websocket_stream()
    
    def get_performance_stats(self) -> Dict:
        """Get performance and reliability statistics."""
        total_data = self.websocket_data_count + self.rest_data_count
        uptime = (datetime.now() - self.last_connection_time).total_seconds() if self.last_connection_time else 0
        
        return {
            'total_data_points': total_data,
            'websocket_percentage': (self.websocket_data_count / total_data * 100) if total_data > 0 else 0,
            'rest_fallback_percentage': (self.rest_data_count / total_data * 100) if total_data > 0 else 0,
            'connection_reliability': max(0, 1 - (self.consecutive_failures / 10)),  # 0-1 scale
            'average_data_rate': total_data / max(uptime / 3600, 0.1),  # per hour
            'connection_uptime': uptime
        }