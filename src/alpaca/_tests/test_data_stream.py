#!/usr/bin/env python3
"""
Unit tests for src/alpaca/data_stream.py - Enhanced Live Data Stream
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading
import time
from collections import deque

from src.alpaca.data_stream import (
    LiveDataStream, DataStreamMode, ConnectionState
)


class TestDataStreamEnums:
    """Test enumeration classes."""
    
    def test_data_stream_mode_values(self):
        """Test DataStreamMode enum values."""
        assert DataStreamMode.WEBSOCKET.value == "websocket"
        assert DataStreamMode.REST_FALLBACK.value == "rest_fallback"
        assert DataStreamMode.OFFLINE.value == "offline"
    
    def test_connection_state_values(self):
        """Test ConnectionState enum values."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.FAILED.value == "failed"


class TestLiveDataStream:
    """Test LiveDataStream class."""
    
    @pytest.fixture
    def mock_historical_client(self):
        """Mock StockHistoricalDataClient."""
        with patch('src.alpaca.data_stream.StockHistoricalDataClient') as mock:
            yield mock
    
    @pytest.fixture
    def mock_stream_client(self):
        """Mock StockDataStream."""
        with patch('src.alpaca.data_stream.StockDataStream') as mock:
            yield mock
    
    @pytest.fixture
    def data_stream(self, mock_historical_client, mock_stream_client):
        """Create test data stream instance."""
        return LiveDataStream(
            buffer_size=10,
            max_retries=3,
            fallback_interval=5
        )
    
    def test_initialization(self, data_stream):
        """Test data stream initialization."""
        assert data_stream.buffer_size == 10
        assert data_stream.max_retries == 3
        assert data_stream.fallback_interval == 5
        assert data_stream.is_running is False
        assert data_stream.current_mode == DataStreamMode.OFFLINE
        assert data_stream.connection_state == ConnectionState.DISCONNECTED
        assert isinstance(data_stream.data_buffers, dict)
        assert isinstance(data_stream.latest_bars, dict)
        assert isinstance(data_stream.subscribers, list)
    
    def test_default_initialization(self, mock_historical_client, mock_stream_client):
        """Test data stream with default parameters."""
        stream = LiveDataStream()
        
        assert stream.buffer_size == 100
        assert stream.max_retries == 5
        assert stream.fallback_interval == 60
    
    def test_add_subscriber(self, data_stream):
        """Test adding subscriber callback."""
        def test_callback(symbol, data):
            pass
        
        data_stream.add_subscriber(test_callback)
        
        assert len(data_stream.subscribers) == 1
        assert data_stream.subscribers[0] == test_callback
    
    def test_multiple_subscribers(self, data_stream):
        """Test adding multiple subscribers."""
        def callback1(symbol, data):
            pass
        
        def callback2(symbol, data):
            pass
        
        data_stream.add_subscriber(callback1)
        data_stream.add_subscriber(callback2)
        
        assert len(data_stream.subscribers) == 2
    
    def test_start_stream_initialization(self, data_stream):
        """Test stream start initialization."""
        symbols = ["SPY", "AAPL"]
        
        with patch.object(data_stream, '_load_initial_data'), \
             patch.object(data_stream, '_start_websocket_stream'), \
             patch.object(data_stream, '_start_connection_monitor'):
            
            data_stream.start_stream(symbols)
        
        assert data_stream.is_running is True
        assert data_stream.symbols == symbols
        assert "SPY" in data_stream.data_buffers
        assert "AAPL" in data_stream.data_buffers
        assert isinstance(data_stream.data_buffers["SPY"], deque)
    
    def test_start_stream_already_running(self, data_stream):
        """Test starting stream when already running."""
        data_stream.is_running = True
        
        with patch('src.alpaca.data_stream.logger') as mock_logger:
            data_stream.start_stream(["SPY"])
            mock_logger.warning.assert_called_with("Data stream already running")
    
    def test_stop_stream(self, data_stream):
        """Test stopping data stream."""
        data_stream.is_running = True
        data_stream.current_mode = DataStreamMode.WEBSOCKET
        
        # Mock threads to avoid actual threading in tests
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        data_stream.stream_thread = mock_thread
        data_stream.fallback_thread = mock_thread
        data_stream.monitor_thread = mock_thread
        
        data_stream.stop_stream()
        
        assert data_stream.is_running is False
        assert data_stream.current_mode == DataStreamMode.OFFLINE
        assert data_stream.connection_state == ConnectionState.DISCONNECTED
    
    def test_stop_stream_with_live_threads(self, data_stream):
        """Test stopping stream with live threads."""
        data_stream.is_running = True
        
        # Mock live threads
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        data_stream.stream_thread = mock_thread
        data_stream.fallback_thread = mock_thread
        data_stream.monitor_thread = mock_thread
        
        data_stream.stop_stream()
        
        # Should call join on all threads
        assert mock_thread.join.call_count == 3
    
    @patch('src.alpaca.data_stream.StockBarsRequest')
    def test_load_initial_data_success(self, mock_request, data_stream):
        """Test successful initial data loading."""
        symbols = ["SPY"]
        
        # Mock historical data response
        mock_bars_response = Mock()
        mock_df = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }])
        mock_bars_response.df = mock_df
        data_stream.historical_client.get_stock_bars.return_value = mock_bars_response
        
        data_stream.symbols = symbols
        data_stream.data_buffers = {"SPY": deque()}
        data_stream.latest_bars = {"SPY": {}}
        
        data_stream._load_initial_data(symbols)
        
        assert len(data_stream.data_buffers["SPY"]) == 1
        assert data_stream.latest_bars["SPY"] is not None
    
    def test_load_initial_data_error(self, data_stream):
        """Test initial data loading with error."""
        symbols = ["SPY"]
        data_stream.historical_client.get_stock_bars.side_effect = Exception("API Error")
        
        with patch('src.alpaca.data_stream.logger') as mock_logger:
            data_stream._load_initial_data(symbols)
            mock_logger.error.assert_called()
    
    def test_format_bar_data(self, data_stream):
        """Test bar data formatting."""
        symbol = "SPY"
        row = {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }
        
        formatted = data_stream._format_bar_data(symbol, row)
        
        assert formatted['symbol'] == symbol
        assert formatted['timestamp'] == row['timestamp']
        assert formatted['open'] == 100.0
        assert formatted['high'] == 101.0
        assert formatted['low'] == 99.0
        assert formatted['close'] == 100.5
        assert formatted['volume'] == 1000
    
    def test_process_bar_data(self, data_stream):
        """Test bar data processing."""
        symbol = "SPY"
        data_stream.data_buffers[symbol] = deque()
        data_stream.latest_bars[symbol] = {}
        
        bar_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }
        
        # Mock subscriber
        mock_subscriber = Mock()
        data_stream.subscribers = [mock_subscriber]
        
        data_stream._process_bar_data(bar_data)
        
        assert len(data_stream.data_buffers[symbol]) == 1
        assert data_stream.latest_bars[symbol] == bar_data
        mock_subscriber.assert_called_once_with(symbol, bar_data)
    
    def test_process_bar_data_subscriber_error(self, data_stream):
        """Test bar data processing with subscriber error."""
        symbol = "SPY"
        data_stream.data_buffers[symbol] = deque()
        
        bar_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'close': 100.5
        }
        
        # Mock failing subscriber
        mock_subscriber = Mock()
        mock_subscriber.side_effect = Exception("Subscriber error")
        data_stream.subscribers = [mock_subscriber]
        
        with patch('src.alpaca.data_stream.logger') as mock_logger:
            data_stream._process_bar_data(bar_data)
            mock_logger.error.assert_called()
    
    def test_get_latest_bar(self, data_stream):
        """Test getting latest bar for symbol."""
        symbol = "SPY"
        bar_data = {'close': 100.5, 'timestamp': datetime.now()}
        data_stream.latest_bars[symbol] = bar_data
        
        latest = data_stream.get_latest_bar(symbol)
        
        assert latest == bar_data
    
    def test_get_latest_bar_not_exists(self, data_stream):
        """Test getting latest bar for non-existent symbol."""
        latest = data_stream.get_latest_bar("NONEXISTENT")
        
        assert latest is None
    
    def test_get_historical_bars(self, data_stream):
        """Test getting historical bars from buffer."""
        symbol = "SPY"
        
        # Add test data to buffer
        data_stream.data_buffers[symbol] = deque()
        for i in range(5):
            bar = {'close': 100.0 + i, 'timestamp': datetime.now()}
            data_stream.data_buffers[symbol].append(bar)
        
        # Get all bars
        bars = data_stream.get_historical_bars(symbol)
        assert len(bars) == 5
        
        # Get limited bars
        bars = data_stream.get_historical_bars(symbol, count=3)
        assert len(bars) == 3
        assert bars[-1]['close'] == 104.0  # Last 3 bars
    
    def test_get_historical_bars_empty(self, data_stream):
        """Test getting historical bars for empty buffer."""
        bars = data_stream.get_historical_bars("NONEXISTENT")
        
        assert bars == []
    
    def test_get_recent_data(self, data_stream):
        """Test getting recent data."""
        symbol = "SPY"
        data_stream.data_buffers[symbol] = deque()
        
        # Add test data
        for i in range(10):
            bar = {'close': 100.0 + i}
            data_stream.data_buffers[symbol].append(bar)
        
        recent = data_stream.get_recent_data(symbol, minutes=5)
        
        assert len(recent) == 5
    
    def test_is_data_fresh_true(self, data_stream):
        """Test data freshness check - fresh data."""
        symbol = "SPY"
        recent_time = datetime.now() - timedelta(minutes=2)
        
        data_stream.latest_bars[symbol] = {
            'timestamp': recent_time
        }
        
        assert data_stream.is_data_fresh(symbol, max_age_minutes=5) is True
    
    def test_is_data_fresh_false(self, data_stream):
        """Test data freshness check - stale data."""
        symbol = "SPY"
        old_time = datetime.now() - timedelta(minutes=10)
        
        data_stream.latest_bars[symbol] = {
            'timestamp': old_time
        }
        
        assert data_stream.is_data_fresh(symbol, max_age_minutes=5) is False
    
    def test_is_data_fresh_no_data(self, data_stream):
        """Test data freshness check - no data."""
        assert data_stream.is_data_fresh("NONEXISTENT") is False
    
    def test_is_data_fresh_invalid_timestamp(self, data_stream):
        """Test data freshness check with invalid timestamp."""
        symbol = "SPY"
        data_stream.latest_bars[symbol] = {
            'timestamp': 'invalid'
        }
        
        assert data_stream.is_data_fresh(symbol) is False
    
    def test_get_ohlcv_dataframe(self, data_stream):
        """Test getting OHLCV DataFrame."""
        symbol = "SPY"
        data_stream.data_buffers[symbol] = deque()
        
        # Add test data
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(3)]
        for i, ts in enumerate(timestamps):
            bar = {
                'timestamp': ts,
                'open': 100.0 + i,
                'high': 101.0 + i,
                'low': 99.0 + i,
                'close': 100.5 + i,
                'volume': 1000 + i
            }
            data_stream.data_buffers[symbol].append(bar)
        
        df = data_stream.get_ohlcv_dataframe(symbol)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        assert list(df.columns) == expected_columns
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_get_ohlcv_dataframe_empty(self, data_stream):
        """Test getting OHLCV DataFrame for empty buffer."""
        df = data_stream.get_ohlcv_dataframe("NONEXISTENT")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_get_current_price(self, data_stream):
        """Test getting current price."""
        symbol = "SPY"
        data_stream.latest_bars[symbol] = {'close': 100.5}
        
        price = data_stream.get_current_price(symbol)
        
        assert price == 100.5
    
    def test_get_current_price_no_data(self, data_stream):
        """Test getting current price with no data."""
        price = data_stream.get_current_price("NONEXISTENT")
        
        assert price is None
    
    def test_get_stream_status(self, data_stream):
        """Test getting stream status."""
        data_stream.is_running = True
        data_stream.current_mode = DataStreamMode.WEBSOCKET
        data_stream.connection_state = ConnectionState.CONNECTED
        data_stream.symbols = ["SPY", "AAPL"]
        data_stream.data_buffers = {"SPY": deque([1, 2, 3])}
        data_stream.websocket_data_count = 100
        data_stream.rest_data_count = 10
        data_stream.last_connection_time = datetime.now()
        
        status = data_stream.get_stream_status()
        
        assert status['is_running'] is True
        assert status['mode'] == "websocket"
        assert status['connection_state'] == "connected"
        assert status['symbols_tracked'] == 2
        assert status['buffer_sizes']['SPY'] == 3
        assert status['websocket_data_count'] == 100
        assert status['rest_data_count'] == 10
    
    def test_get_performance_stats(self, data_stream):
        """Test getting performance statistics."""
        data_stream.websocket_data_count = 80
        data_stream.rest_data_count = 20
        data_stream.consecutive_failures = 2
        data_stream.last_connection_time = datetime.now() - timedelta(hours=1)
        
        stats = data_stream.get_performance_stats()
        
        assert stats['total_data_points'] == 100
        assert stats['websocket_percentage'] == 80.0
        assert stats['rest_fallback_percentage'] == 20.0
        assert 0 <= stats['connection_reliability'] <= 1
        assert stats['average_data_rate'] > 0
    
    def test_get_performance_stats_no_data(self, data_stream):
        """Test performance stats with no data."""
        data_stream.websocket_data_count = 0
        data_stream.rest_data_count = 0
        
        stats = data_stream.get_performance_stats()
        
        assert stats['total_data_points'] == 0
        assert stats['websocket_percentage'] == 0
        assert stats['rest_fallback_percentage'] == 0
    
    def test_force_websocket_recovery(self, data_stream):
        """Test manual WebSocket recovery trigger."""
        data_stream.current_mode = DataStreamMode.REST_FALLBACK
        data_stream.retry_count = 5
        
        with patch.object(data_stream, '_start_websocket_stream') as mock_start:
            data_stream.force_websocket_recovery()
            
            assert data_stream.retry_count == 0
            mock_start.assert_called_once()
    
    def test_force_websocket_recovery_not_in_fallback(self, data_stream):
        """Test WebSocket recovery when not in fallback mode."""
        data_stream.current_mode = DataStreamMode.WEBSOCKET
        
        with patch.object(data_stream, '_start_websocket_stream') as mock_start:
            data_stream.force_websocket_recovery()
            
            # Should not start WebSocket when already in WebSocket mode
            mock_start.assert_not_called()
    
    def test_websocket_bar_handler(self, data_stream):
        """Test WebSocket bar handler."""
        # Mock bar object
        mock_bar = Mock()
        mock_bar.symbol = "SPY"
        mock_bar.timestamp = datetime.now()
        mock_bar.open = 100.0
        mock_bar.high = 101.0
        mock_bar.low = 99.0
        mock_bar.close = 100.5
        mock_bar.volume = 1000
        
        data_stream.data_buffers["SPY"] = deque()
        data_stream.latest_bars["SPY"] = {}
        
        # Run async handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(data_stream._handle_websocket_bar(mock_bar))
            
            assert data_stream.websocket_data_count == 1
            assert len(data_stream.data_buffers["SPY"]) == 1
        finally:
            loop.close()
    
    def test_websocket_bar_handler_error(self, data_stream):
        """Test WebSocket bar handler with error."""
        # Mock bar with missing attributes
        mock_bar = Mock()
        mock_bar.symbol = "SPY"
        del mock_bar.timestamp  # This will cause an AttributeError
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            with patch('src.alpaca.data_stream.logger') as mock_logger:
                loop.run_until_complete(data_stream._handle_websocket_bar(mock_bar))
                mock_logger.error.assert_called()
        finally:
            loop.close()


class TestDataStreamIntegration:
    """Integration tests for data stream functionality."""
    
    @pytest.fixture
    def stream_with_mocks(self):
        """Create data stream with mocked dependencies."""
        with patch('src.alpaca.data_stream.StockHistoricalDataClient'), \
             patch('src.alpaca.data_stream.StockDataStream'):
            stream = LiveDataStream(buffer_size=5, max_retries=2, fallback_interval=1)
            yield stream
    
    def test_full_lifecycle(self, stream_with_mocks):
        """Test complete data stream lifecycle."""
        symbols = ["SPY"]
        
        with patch.object(stream_with_mocks, '_load_initial_data'), \
             patch.object(stream_with_mocks, '_start_websocket_stream'), \
             patch.object(stream_with_mocks, '_start_connection_monitor'):
            
            # Start stream
            stream_with_mocks.start_stream(symbols)
            assert stream_with_mocks.is_running is True
            
            # Simulate data reception
            bar_data = {
                'symbol': 'SPY',
                'timestamp': datetime.now(),
                'close': 100.5
            }
            stream_with_mocks._process_bar_data(bar_data)
            
            # Check data storage
            assert stream_with_mocks.get_latest_bar('SPY') is not None
            assert stream_with_mocks.get_current_price('SPY') == 100.5
            
            # Stop stream
            stream_with_mocks.stop_stream()
            assert stream_with_mocks.is_running is False


if __name__ == "__main__":
    pytest.main([__file__])