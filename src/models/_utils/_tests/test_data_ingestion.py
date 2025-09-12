#!/usr/bin/env python3
"""
Unit tests for src/models/_utils/data_ingestion.py - Stock Data Ingestion
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import os
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.models._utils.data_ingestion import fetch_stock_data


class TestFetchStockData:
    """Test fetch_stock_data function."""
    
    @pytest.fixture
    def mock_historical_client(self):
        """Mock StockHistoricalDataClient."""
        with patch('src.models._utils.data_ingestion.StockHistoricalDataClient') as mock:
            yield mock
    
    @pytest.fixture
    def sample_bars_response(self):
        """Create sample bars response."""
        # Create sample data
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=10, freq='1min')
        data = []
        
        for i, ts in enumerate(timestamps):
            data.append({
                'symbol': 'SPY',
                'timestamp': ts,
                'open': 100.0 + i * 0.1,
                'high': 101.0 + i * 0.1,
                'low': 99.0 + i * 0.1,
                'close': 100.5 + i * 0.1,
                'volume': 1000 + i * 10
            })
        
        df = pd.DataFrame(data)
        df = df.set_index(['symbol', 'timestamp'])
        
        mock_response = Mock()
        mock_response.df = df
        return mock_response
    
    def test_fetch_stock_data_success(self, mock_historical_client, sample_bars_response):
        """Test successful stock data fetching."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.return_value = sample_bars_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.object(sample_bars_response.df, 'to_parquet') as mock_to_parquet:
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Verify client was created with correct credentials
            mock_historical_client.assert_called_once()
            
            # Verify API call was made
            mock_client_instance.get_stock_bars.assert_called_once()
            
            # Verify request parameters
            call_args = mock_client_instance.get_stock_bars.call_args[0][0]
            assert isinstance(call_args, StockBarsRequest)
            
            # Verify data was saved
            mock_to_parquet.assert_called_once()
    
    def test_fetch_stock_data_file_exists(self, mock_historical_client):
        """Test fetch_stock_data when file already exists (caching)."""
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=True), \
             patch('src.models._utils.data_ingestion.logger') as mock_logger:
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Should not create client when file exists
            mock_historical_client.assert_not_called()
            
            # Should log cache hit
            mock_logger.info.assert_any_call("Data for SPY already exists locally. Skipping download.")
    
    def test_fetch_stock_data_multiple_tickers(self, mock_historical_client, sample_bars_response):
        """Test fetching data for multiple tickers."""
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.return_value = sample_bars_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY", "AAPL", "MSFT"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.object(sample_bars_response.df, 'to_parquet'):
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Should make API call for each ticker
            assert mock_client_instance.get_stock_bars.call_count == len(tickers)
    
    def test_fetch_stock_data_mixed_cache_status(self, mock_historical_client, sample_bars_response):
        """Test fetching when some files exist and others don't."""
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.return_value = sample_bars_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY", "AAPL"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        # Mock file existence: SPY exists, AAPL doesn't
        def mock_exists(path):
            return "SPY" in path
        
        with patch('os.path.exists', side_effect=mock_exists), \
             patch('os.makedirs'), \
             patch.object(sample_bars_response.df, 'to_parquet'):
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Should only fetch AAPL (SPY is cached)
            assert mock_client_instance.get_stock_bars.call_count == 1
    
    def test_fetch_stock_data_empty_response(self, mock_historical_client):
        """Test handling of empty API response."""
        mock_client_instance = Mock()
        
        # Mock empty response
        empty_response = Mock()
        empty_response.df = pd.DataFrame()  # Empty DataFrame
        mock_client_instance.get_stock_bars.return_value = empty_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["INVALID"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('src.models._utils.data_ingestion.logger') as mock_logger:
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Should log warning about no data
            mock_logger.warning.assert_called_with(
                "No data returned for INVALID for the given date range."
            )
    
    def test_fetch_stock_data_api_error(self, mock_historical_client):
        """Test handling of API errors."""
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.side_effect = Exception("API Error")
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('src.models._utils.data_ingestion.logger') as mock_logger:
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Should log error
            mock_logger.error.assert_called_with("Failed to fetch data for SPY: API Error")
    
    def test_fetch_stock_data_date_parsing(self, mock_historical_client, sample_bars_response):
        """Test date string parsing."""
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.return_value = sample_bars_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.object(sample_bars_response.df, 'to_parquet'):
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Verify request was made with correct datetime objects
            call_args = mock_client_instance.get_stock_bars.call_args[0][0]
            assert call_args.start == datetime(2024, 1, 1)
            assert call_args.end == datetime(2024, 12, 31)
    
    def test_fetch_stock_data_request_parameters(self, mock_historical_client, sample_bars_response):
        """Test that request is created with correct parameters."""
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.return_value = sample_bars_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.object(sample_bars_response.df, 'to_parquet'):
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Check request parameters
            call_args = mock_client_instance.get_stock_bars.call_args[0][0]
            assert call_args.symbol_or_symbols == ["SPY"]
            assert call_args.timeframe == TimeFrame.Minute
            assert call_args.feed == "sip"
    
    def test_fetch_stock_data_data_processing(self, mock_historical_client):
        """Test data processing and formatting."""
        mock_client_instance = Mock()
        
        # Create raw response data
        raw_data = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': '2024-01-01 09:30:00',
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }])
        
        mock_response = Mock()
        mock_response.df = raw_data
        mock_client_instance.get_stock_bars.return_value = mock_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs, \
             patch.object(raw_data, 'to_parquet') as mock_to_parquet:
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Verify directory creation
            mock_makedirs.assert_called_once()
            
            # Verify data was processed and saved
            mock_to_parquet.assert_called_once()
    
    def test_fetch_stock_data_file_path_construction(self, mock_historical_client, sample_bars_response):
        """Test correct file path construction."""
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.return_value = sample_bars_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.object(sample_bars_response.df, 'to_parquet') as mock_to_parquet:
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Check the file path used for saving
            expected_path = os.path.join("data", "SPY_1min_2024-01-01_to_2024-01-02.parquet")
            mock_to_parquet.assert_called_with(expected_path)
    
    def test_fetch_stock_data_invalid_date_format(self, mock_historical_client):
        """Test handling of invalid date formats."""
        tickers = ["SPY"]
        start_date = "invalid-date"
        end_date = "2024-01-02"
        
        with pytest.raises(ValueError):
            fetch_stock_data(tickers, start_date, end_date)
    
    def test_fetch_stock_data_timestamp_processing(self, mock_historical_client):
        """Test timestamp processing and timezone handling."""
        mock_client_instance = Mock()
        
        # Create data with timezone-aware timestamps
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=2, freq='1min', tz='UTC')
        raw_data = pd.DataFrame({
            'symbol': ['SPY', 'SPY'],
            'timestamp': timestamps,
            'open': [100.0, 100.1],
            'high': [101.0, 101.1],
            'low': [99.0, 99.1],
            'close': [100.5, 100.6],
            'volume': [1000, 1001]
        })
        
        mock_response = Mock()
        mock_response.df = raw_data
        mock_client_instance.get_stock_bars.return_value = mock_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.object(raw_data, 'to_parquet'):
            
            # Should not raise error with timezone-aware timestamps
            fetch_stock_data(tickers, start_date, end_date)
    
    def test_fetch_stock_data_empty_ticker_list(self, mock_historical_client):
        """Test handling of empty ticker list."""
        tickers = []
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        fetch_stock_data(tickers, start_date, end_date)
        
        # Should not create client for empty ticker list
        mock_historical_client.assert_not_called()
    
    @patch('src.models._utils.data_ingestion.ALPACA_API_KEY', 'test_key')
    @patch('src.models._utils.data_ingestion.ALPACA_SECRET_KEY', 'test_secret')
    def test_fetch_stock_data_credentials(self, mock_historical_client, sample_bars_response):
        """Test that correct credentials are used."""
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.return_value = sample_bars_response
        mock_historical_client.return_value = mock_client_instance
        
        tickers = ["SPY"]
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.object(sample_bars_response.df, 'to_parquet'):
            
            fetch_stock_data(tickers, start_date, end_date)
            
            # Verify client was created with correct credentials
            mock_historical_client.assert_called_with('test_key', 'test_secret')


class TestDataIngestionIntegration:
    """Integration tests for data ingestion functionality."""
    
    def test_data_ingestion_pipeline(self):
        """Test complete data ingestion pipeline with mocked components."""
        # Create realistic mock data
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=5, freq='1min')
        mock_data = pd.DataFrame({
            'symbol': ['SPY'] * 5,
            'timestamp': timestamps,
            'open': [100.0, 100.1, 100.2, 100.3, 100.4],
            'high': [100.5, 100.6, 100.7, 100.8, 100.9],
            'low': [99.5, 99.6, 99.7, 99.8, 99.9],
            'close': [100.2, 100.3, 100.4, 100.5, 100.6],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        mock_data = mock_data.set_index(['symbol', 'timestamp'])
        
        with patch('src.models._utils.data_ingestion.StockHistoricalDataClient') as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.df = mock_data
            mock_client.get_stock_bars.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            with patch('os.path.exists', return_value=False), \
                 patch('os.makedirs') as mock_makedirs, \
                 patch.object(mock_data, 'to_parquet') as mock_save:
                
                # Run the function
                fetch_stock_data(['SPY'], '2024-01-01', '2024-01-01')
                
                # Verify complete pipeline
                mock_client_class.assert_called_once()
                mock_client.get_stock_bars.assert_called_once()
                mock_makedirs.assert_called_once()
                mock_save.assert_called_once()
    
    def test_error_recovery_across_tickers(self):
        """Test that errors for one ticker don't affect others."""
        with patch('src.models._utils.data_ingestion.StockHistoricalDataClient') as mock_client_class:
            mock_client = Mock()
            
            # Mock different responses for different tickers
            def mock_get_bars(request):
                if 'SPY' in str(request.symbol_or_symbols):
                    raise Exception("SPY API Error")
                else:
                    # Return valid data for other tickers
                    mock_response = Mock()
                    mock_response.df = pd.DataFrame([{
                        'symbol': 'AAPL',
                        'timestamp': datetime.now(),
                        'open': 150.0,
                        'high': 151.0,
                        'low': 149.0,
                        'close': 150.5,
                        'volume': 2000
                    }]).set_index(['symbol', 'timestamp'])
                    return mock_response
            
            mock_client.get_stock_bars.side_effect = mock_get_bars
            mock_client_class.return_value = mock_client
            
            with patch('os.path.exists', return_value=False), \
                 patch('os.makedirs'), \
                 patch('src.models._utils.data_ingestion.logger') as mock_logger:
                
                # Should continue processing AAPL despite SPY error
                fetch_stock_data(['SPY', 'AAPL'], '2024-01-01', '2024-01-01')
                
                # Should log error for SPY but continue with AAPL
                mock_logger.error.assert_called()
                
                # Should make calls for both tickers
                assert mock_client.get_stock_bars.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])