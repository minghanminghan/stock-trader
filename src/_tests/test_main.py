#!/usr/bin/env python3
"""
Unit tests for src/main.py - Trading System Orchestrator
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import threading
import time

from src.main import (
    TradingOrchestrator, 
    OrchestratorConfig, 
    SystemHealth
)


class TestOrchestratorConfig:
    """Test OrchestratorConfig dataclass."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        
        assert isinstance(config.symbols, list)
        assert config.paper_trading is True
        assert config.signal_generation_interval == 60
        assert config.min_data_points == 60
        assert config.health_check_interval == 30
        assert config.max_consecutive_failures == 5
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        custom_symbols = ["SPY", "AAPL"]
        config = OrchestratorConfig(
            symbols=custom_symbols,
            paper_trading=False,
            signal_generation_interval=30
        )
        
        assert config.symbols == custom_symbols
        assert config.paper_trading is False
        assert config.signal_generation_interval == 30


class TestSystemHealth:
    """Test SystemHealth dataclass."""
    
    def test_default_health_state(self):
        """Test default health state."""
        health = SystemHealth()
        
        assert health.data_stream_healthy is False
        assert health.broker_healthy is False
        assert health.signal_generator_healthy is False
        assert health.consecutive_failures == 0
        assert health.total_signals_generated == 0


class TestTradingOrchestrator:
    """Test TradingOrchestrator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OrchestratorConfig(
            symbols=["SPY", "AAPL"],
            signal_generation_interval=1,  # Fast for testing
            health_check_interval=1,
            min_data_points=5
        )
    
    @pytest.fixture
    def orchestrator(self, config):
        """Create test orchestrator."""
        return TradingOrchestrator(config)
    
    def test_init(self, orchestrator, config):
        """Test orchestrator initialization."""
        assert orchestrator.config == config
        assert orchestrator.running is False
        assert isinstance(orchestrator.health, SystemHealth)
        assert orchestrator.latest_market_data == {}
        assert isinstance(orchestrator.stats, dict)
    
    def test_data_callback_processing(self, orchestrator):
        """Test processing of new data callback."""
        # Mock data
        test_symbol = "SPY"
        test_bar = {
            'timestamp': datetime.now(timezone.utc),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }
        
        # Process data
        orchestrator._on_new_data(test_symbol, test_bar)
        
        # Verify data storage
        assert test_symbol in orchestrator.latest_market_data
        assert len(orchestrator.latest_market_data[test_symbol]) == 1
        assert orchestrator.stats['data_points_processed'] == 1
        assert orchestrator.health.last_data_received is not None
    
    def test_data_buffer_size_limit(self, orchestrator):
        """Test data buffer respects size limits."""
        test_symbol = "SPY"
        current_time = datetime.now(timezone.utc)
        
        # Add more data than buffer allows
        for i in range(orchestrator.config.data_buffer_minutes + 10):
            test_bar = {
                'timestamp': current_time - timedelta(minutes=i),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000
            }
            orchestrator._on_new_data(test_symbol, test_bar)
        
        # Should only keep recent data within buffer window
        assert len(orchestrator.latest_market_data[test_symbol]) <= orchestrator.config.data_buffer_minutes
    
    def test_sufficient_data_check(self, orchestrator):
        """Test sufficient data checking."""
        # Initially no data
        assert orchestrator._has_sufficient_data() is False
        
        # Add insufficient data
        for symbol in orchestrator.config.symbols:
            for i in range(orchestrator.config.min_data_points - 1):
                test_bar = {
                    'timestamp': datetime.now(timezone.utc),
                    'open': 100.0,
                    'high': 101.0,
                    'low': 99.0,
                    'close': 100.5,
                    'volume': 1000
                }
                orchestrator._on_new_data(symbol, test_bar)
        
        assert orchestrator._has_sufficient_data() is False
        
        # Add one more data point to reach minimum
        for symbol in orchestrator.config.symbols:
            test_bar = {
                'timestamp': datetime.now(timezone.utc),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000
            }
            orchestrator._on_new_data(symbol, test_bar)
        
        assert orchestrator._has_sufficient_data() is True
    
    def test_market_data_preparation(self, orchestrator):
        """Test market data preparation for signal generation."""
        # Add test data
        for symbol in orchestrator.config.symbols:
            for i in range(10):
                test_bar = {
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'open': 100.0 + i,
                    'high': 101.0 + i,
                    'low': 99.0 + i,
                    'close': 100.5 + i,
                    'volume': 1000 + i
                }
                orchestrator._on_new_data(symbol, test_bar)
        
        # Prepare market data
        market_data = orchestrator._prepare_market_data()
        
        assert market_data is not None
        assert isinstance(market_data, pd.DataFrame)
        assert len(market_data.index.get_level_values(0).unique()) == len(orchestrator.config.symbols)
    
    def test_market_data_preparation_empty(self, orchestrator):
        """Test market data preparation with no data."""
        market_data = orchestrator._prepare_market_data()
        assert market_data is None
    
    @patch('src.main.compute_features')
    def test_market_data_preparation_with_feature_error(self, mock_compute_features, orchestrator):
        """Test market data preparation when feature computation fails."""
        mock_compute_features.side_effect = Exception("Feature computation failed")
        
        # Add test data
        for symbol in orchestrator.config.symbols:
            test_bar = {
                'timestamp': datetime.now(timezone.utc),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000
            }
            orchestrator._on_new_data(symbol, test_bar)
        
        market_data = orchestrator._prepare_market_data()
        assert market_data is not None  # Should return raw data without features
    
    @patch('src.main.LiveDataStream')
    @patch('src.main.AlpacaBroker')
    @patch('src.main.create_signal_generator')
    def test_component_initialization_success(self, mock_signal_gen, mock_broker, mock_stream, orchestrator):
        """Test successful component initialization."""
        # Mock successful initialization
        mock_stream.return_value = Mock()
        mock_broker.return_value = Mock()
        mock_broker.return_value.get_account.return_value = {'portfolio_value': 10000.0}
        mock_signal_gen.return_value = Mock()
        
        result = orchestrator.initialize_components()
        
        assert result is True
        assert orchestrator.health.broker_healthy is True
        assert orchestrator.health.signal_generator_healthy is True
    
    @patch('src.main.LiveDataStream')
    def test_component_initialization_failure(self, mock_stream, orchestrator):
        """Test component initialization failure."""
        # Mock initialization failure
        mock_stream.side_effect = Exception("Initialization failed")
        
        result = orchestrator.initialize_components()
        
        assert result is False
    
    def test_health_check_data_staleness(self, orchestrator):
        """Test health check for stale data."""
        # Set old data received time
        orchestrator.health.last_data_received = datetime.now(timezone.utc) - timedelta(
            seconds=orchestrator.config.data_staleness_threshold + 60
        )
        
        with patch('src.main.logger') as mock_logger:
            orchestrator._check_system_health()
            mock_logger.warning.assert_called()
    
    @patch('src.main.create_signal_generator')
    def test_component_recovery(self, mock_signal_gen, orchestrator):
        """Test component recovery attempt."""
        mock_signal_gen.return_value = Mock()
        orchestrator.broker = Mock()
        orchestrator.config.strategy_config = Mock()
        
        orchestrator._attempt_component_recovery()
        
        assert orchestrator.health.consecutive_failures == 0
        assert orchestrator.stats['component_restarts'] == 1
    
    def test_component_recovery_failure(self, orchestrator):
        """Test component recovery failure."""
        orchestrator.broker = None  # Force failure
        
        orchestrator._attempt_component_recovery()
        
        assert orchestrator.stats['degraded_operations'] == 1
    
    def test_statistics_update(self, orchestrator):
        """Test statistics updating."""
        orchestrator.start_time = datetime.now(timezone.utc) - timedelta(seconds=3600)
        
        orchestrator._update_statistics()
        
        assert orchestrator.stats['uptime_seconds'] > 3500  # Approximately 1 hour
    
    def test_data_callback_error_handling(self, orchestrator):
        """Test error handling in data callback."""
        # Invalid data should not crash
        with patch('src.main.logger') as mock_logger:
            orchestrator._on_new_data("SPY", {"invalid": "data"})
            mock_logger.error.assert_called()
    
    def test_stop_before_start(self, orchestrator):
        """Test stopping orchestrator before starting."""
        # Should not raise error
        orchestrator.stop()
        assert orchestrator.running is False
    
    def test_statistics_initialization(self, orchestrator):
        """Test initial statistics state."""
        expected_keys = {
            'uptime_seconds', 'data_points_processed', 'signals_generated',
            'orders_executed', 'component_restarts', 'degraded_operations'
        }
        
        assert set(orchestrator.stats.keys()) == expected_keys
        assert all(isinstance(v, (int, float)) for v in orchestrator.stats.values())


class TestTradingOrchestratorIntegration:
    """Integration tests for TradingOrchestrator."""
    
    @pytest.fixture
    def mock_config(self):
        """Create configuration for integration tests."""
        return OrchestratorConfig(
            symbols=["TEST"],
            signal_generation_interval=0.1,  # Very fast
            health_check_interval=0.1,
            min_data_points=1
        )
    
    @patch('src.main.LiveDataStream')
    @patch('src.main.AlpacaBroker')
    @patch('src.main.create_signal_generator')
    def test_orchestrator_lifecycle(self, mock_signal_gen, mock_broker, mock_stream, mock_config):
        """Test complete orchestrator lifecycle."""
        # Setup mocks
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance
        
        mock_broker_instance = Mock()
        mock_broker_instance.get_account.return_value = {'portfolio_value': 10000.0}
        mock_broker.return_value = mock_broker_instance
        
        mock_signal_gen.return_value = Mock()
        
        orchestrator = TradingOrchestrator(mock_config)
        
        # Initialize components
        assert orchestrator.initialize_components() is True
        
        # Verify components are set
        assert hasattr(orchestrator, 'data_stream')
        assert hasattr(orchestrator, 'broker')
        assert hasattr(orchestrator, 'signal_generator')
        
        # Test data processing
        test_bar = {
            'timestamp': datetime.now(timezone.utc),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }
        orchestrator._on_new_data("TEST", test_bar)
        
        assert orchestrator._has_sufficient_data() is True
        
        # Test market data preparation
        market_data = orchestrator._prepare_market_data()
        assert market_data is not None


def test_main_function():
    """Test main function structure."""
    from src.main import main
    
    # Should be callable without error (though will fail due to missing dependencies)
    assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__])