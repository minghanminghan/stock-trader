#!/usr/bin/env python3
"""
Unit tests for src/trading/signal_generation.py - Integrated Signal Generation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from src.trading.signal_generation import (
    PredictionAdapter, SignalGenerator, create_signal_generator
)


class TestPredictionAdapter:
    """Test PredictionAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create test prediction adapter."""
        horizon_weights = {
            '5min': 2.0,
            '15min': 1.5,
            '30min': 1.0,
            '60min': 0.8
        }
        return PredictionAdapter(
            horizon_weights=horizon_weights,
            min_return_threshold=0.001
        )
    
    @pytest.fixture
    def sample_lstm_predictions(self):
        """Create sample LSTM predictions."""
        return {
            'SPY': {
                '5min': {'price': 101.0, 'confidence': 0.8, 'variance': 1.0},
                '15min': {'price': 101.5, 'confidence': 0.7, 'variance': 1.5},
                '30min': {'price': 102.0, 'confidence': 0.6, 'variance': 2.0},
                '60min': {'price': 102.5, 'confidence': 0.5, 'variance': 2.5}
            },
            'AAPL': {
                '5min': {'price': 149.0, 'confidence': 0.9, 'variance': 0.8},
                '15min': {'price': 148.5, 'confidence': 0.8, 'variance': 1.2},
                '30min': {'price': 148.0, 'confidence': 0.7, 'variance': 1.8},
                '60min': {'price': 147.5, 'confidence': 0.6, 'variance': 2.2}
            }
        }
    
    @pytest.fixture
    def current_prices(self):
        """Create current prices."""
        return {
            'SPY': 100.0,
            'AAPL': 150.0
        }
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.horizon_weights == {
            '5min': 2.0, '15min': 1.5, '30min': 1.0, '60min': 0.8
        }
        assert adapter.min_return_threshold == 0.001
    
    def test_convert_predictions_basic(self, adapter, sample_lstm_predictions, current_prices):
        """Test basic prediction conversion."""
        result = adapter.convert_predictions(sample_lstm_predictions, current_prices)
        
        assert isinstance(result, dict)
        assert 'SPY' in result
        assert 'AAPL' in result
        
        # Check SPY prediction structure
        spy_pred = result['SPY']
        expected_keys = {'direction', 'confidence', 'expected_return', 'current_price'}
        assert set(spy_pred.keys()) == expected_keys
    
    def test_convert_predictions_direction_up(self, adapter, current_prices):
        """Test prediction conversion for upward direction."""
        lstm_predictions = {
            'SPY': {
                '5min': {'price': 101.0, 'confidence': 0.8},  # 1% up
                '15min': {'price': 101.5, 'confidence': 0.7}   # 1.5% up
            }
        }
        
        result = adapter.convert_predictions(lstm_predictions, current_prices)
        
        spy_pred = result['SPY']
        assert spy_pred['direction'] == 'up'
        assert spy_pred['expected_return'] > adapter.min_return_threshold
    
    def test_convert_predictions_direction_down(self, adapter, current_prices):
        """Test prediction conversion for downward direction."""
        lstm_predictions = {
            'SPY': {
                '5min': {'price': 99.0, 'confidence': 0.8},   # 1% down
                '15min': {'price': 98.5, 'confidence': 0.7}   # 1.5% down
            }
        }
        
        result = adapter.convert_predictions(lstm_predictions, current_prices)
        
        spy_pred = result['SPY']
        assert spy_pred['direction'] == 'down'
        assert spy_pred['expected_return'] < -adapter.min_return_threshold
    
    def test_convert_predictions_direction_hold(self, adapter, current_prices):
        """Test prediction conversion for hold direction."""
        lstm_predictions = {
            'SPY': {
                '5min': {'price': 100.05, 'confidence': 0.8},  # 0.05% up (below threshold)
                '15min': {'price': 99.95, 'confidence': 0.7}   # 0.05% down (below threshold)
            }
        }
        
        result = adapter.convert_predictions(lstm_predictions, current_prices)
        
        spy_pred = result['SPY']
        assert spy_pred['direction'] == 'hold'
        assert abs(spy_pred['expected_return']) < adapter.min_return_threshold
    
    def test_convert_predictions_weighted_average(self, adapter, current_prices):
        """Test that predictions are properly weighted across horizons."""
        lstm_predictions = {
            'SPY': {
                '5min': {'price': 102.0, 'confidence': 0.8},   # 2% up, weight 2.0
                '15min': {'price': 98.0, 'confidence': 0.6}    # 2% down, weight 1.5
            }
        }
        
        result = adapter.convert_predictions(lstm_predictions, current_prices)
        
        spy_pred = result['SPY']
        
        # Weighted expected return should favor 5min (higher weight)
        # Expected: (2.0 * 0.02 + 1.5 * (-0.02)) / (2.0 + 1.5) = 0.01 / 3.5 ≈ 0.0029
        expected_weighted_return = (2.0 * 0.02 + 1.5 * (-0.02)) / (2.0 + 1.5)
        assert abs(spy_pred['expected_return'] - expected_weighted_return) < 1e-6
    
    def test_convert_predictions_empty_predictions(self, adapter, current_prices):
        """Test conversion with empty predictions."""
        result = adapter.convert_predictions({}, current_prices)
        
        assert result == {}
    
    def test_convert_predictions_missing_symbol_price(self, adapter, sample_lstm_predictions):
        """Test conversion when current price is missing."""
        current_prices = {'SPY': 100.0}  # Missing AAPL
        
        result = adapter.convert_predictions(sample_lstm_predictions, current_prices)
        
        # Should only have SPY (AAPL skipped due to missing price)
        assert 'SPY' in result
        assert 'AAPL' not in result
    
    def test_convert_predictions_zero_current_price(self, adapter, sample_lstm_predictions):
        """Test conversion with zero current price."""
        current_prices = {'SPY': 0.0, 'AAPL': 150.0}
        
        result = adapter.convert_predictions(sample_lstm_predictions, current_prices)
        
        # Should skip SPY (zero price) but include AAPL
        assert 'SPY' not in result
        assert 'AAPL' in result
    
    def test_convert_predictions_missing_horizon_weights(self, adapter, current_prices):
        """Test conversion with horizons not in weights."""
        lstm_predictions = {
            'SPY': {
                '2min': {'price': 101.0, 'confidence': 0.8},   # Not in weights
                '5min': {'price': 101.5, 'confidence': 0.7}    # In weights
            }
        }
        
        result = adapter.convert_predictions(lstm_predictions, current_prices)
        
        # Should only use 5min prediction (2min not in weights)
        assert 'SPY' in result
        spy_pred = result['SPY']
        
        # Expected return should be based only on 5min prediction
        expected_return = (101.5 - 100.0) / 100.0
        assert abs(spy_pred['expected_return'] - expected_return) < 1e-6


class TestSignalGenerator:
    """Test SignalGenerator class."""
    
    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        return Mock()
    
    @pytest.fixture
    def mock_strategy_config(self):
        """Create mock strategy configuration."""
        mock_config = Mock()
        mock_config.symbols = ['SPY', 'AAPL']
        return mock_config
    
    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy."""
        strategy = Mock()
        strategy.generate_signals.return_value = {
            'SPY': {'action': 'buy', 'confidence': 0.8, 'quantity': 10},
            'AAPL': {'action': 'sell', 'confidence': 0.7, 'quantity': 5}
        }
        strategy.process_signals.return_value = [
            {'symbol': 'SPY', 'side': 'buy', 'quantity': 10},
            {'symbol': 'AAPL', 'side': 'sell', 'quantity': 5}
        ]
        return strategy
    
    @pytest.fixture
    def mock_predictor(self):
        """Create mock LSTM predictor."""
        predictor = Mock()
        predictor.predict_prices.return_value = {
            'SPY': {'5min': {'price': 101.0, 'confidence': 0.8}},
            'AAPL': {'5min': {'price': 149.0, 'confidence': 0.9}}
        }
        return predictor
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=10, freq='1min')
        data = []
        
        for symbol in ['SPY', 'AAPL']:
            for i, ts in enumerate(timestamps):
                data.append({
                    'symbol': symbol,
                    'timestamp': ts,
                    'open': 100.0 + i * 0.1,
                    'high': 101.0 + i * 0.1,
                    'low': 99.0 + i * 0.1,
                    'close': 100.5 + i * 0.1,
                    'volume': 1000 + i
                })
        
        df = pd.DataFrame(data)
        return df.set_index(['symbol', 'timestamp'])
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_signal_generator_initialization(self, mock_predictor_class, mock_create_strategy,
                                           mock_broker, mock_strategy_config):
        """Test signal generator initialization."""
        mock_strategy = Mock()
        mock_create_strategy.return_value = mock_strategy
        mock_predictor_instance = Mock()
        mock_predictor_class.return_value = mock_predictor_instance
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        assert generator.broker == mock_broker
        assert generator.strategy_config == mock_strategy_config
        assert generator.strategy == mock_strategy
        assert generator.predictor == mock_predictor_instance
        assert isinstance(generator.adapter, PredictionAdapter)
        assert isinstance(generator.recent_signals, dict)
        assert isinstance(generator.execution_stats, dict)
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_signal_generator_predictor_failure(self, mock_predictor_class, mock_create_strategy,
                                              mock_broker, mock_strategy_config):
        """Test signal generator when predictor initialization fails."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.side_effect = Exception("Model loading failed")
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="invalid_model.pth"
        )
        
        # Should handle predictor failure gracefully
        assert generator.predictor is None
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_generate_and_execute_signals_success(self, mock_predictor_class, mock_create_strategy,
                                                mock_broker, mock_strategy_config, 
                                                sample_market_data):
        """Test successful signal generation and execution."""
        # Setup mocks
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = {
            'SPY': {'action': 'buy', 'confidence': 0.8}
        }
        mock_strategy.process_signals.return_value = [
            {'symbol': 'SPY', 'side': 'buy', 'quantity': 10}
        ]
        mock_create_strategy.return_value = mock_strategy
        
        mock_predictor = Mock()
        mock_predictor.predict_prices.return_value = {
            'SPY': {'5min': {'price': 101.0, 'confidence': 0.8}},
            'AAPL': {'5min': {'price': 149.0, 'confidence': 0.9}}
        }
        mock_predictor_class.return_value = mock_predictor
        
        # Mock successful order execution
        mock_order_result = {'success': True, 'order_id': '12345'}
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        with patch.object(generator, '_execute_order', return_value=mock_order_result):
            result = generator.generate_and_execute_signals(sample_market_data)
        
        # Check result structure
        expected_keys = {
            'timestamp', 'symbols_processed', 'signals_generated', 
            'orders_placed', 'orders_successful', 'errors',
            'raw_predictions', 'adapted_predictions', 
            'strategy_signals', 'orders'
        }
        assert set(result.keys()) == expected_keys
        
        # Check that pipeline was executed
        assert result['symbols_processed'] > 0
        assert result['orders_placed'] > 0
        assert result['orders_successful'] > 0
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_generate_and_execute_signals_no_predictor(self, mock_predictor_class, mock_create_strategy,
                                                     mock_broker, mock_strategy_config, 
                                                     sample_market_data):
        """Test signal generation when predictor is unavailable."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.side_effect = Exception("Model failed")
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="invalid_model.pth"
        )
        
        result = generator.generate_and_execute_signals(sample_market_data)
        
        # Should handle gracefully
        assert result['symbols_processed'] > 0
        assert "LSTM predictor not available" in result['errors']
        assert result['orders_placed'] == 0
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_extract_current_prices(self, mock_predictor_class, mock_create_strategy,
                                  mock_broker, mock_strategy_config, sample_market_data):
        """Test current price extraction from market data."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.return_value = Mock()
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        prices = generator._extract_current_prices(sample_market_data)
        
        assert isinstance(prices, dict)
        assert 'SPY' in prices
        assert 'AAPL' in prices
        assert all(isinstance(price, float) for price in prices.values())
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_extract_current_prices_empty_data(self, mock_predictor_class, mock_create_strategy,
                                             mock_broker, mock_strategy_config):
        """Test current price extraction with empty data."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.return_value = Mock()
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        empty_df = pd.DataFrame()
        empty_df.index = pd.MultiIndex.from_tuples([], names=['symbol', 'timestamp'])
        
        prices = generator._extract_current_prices(empty_df)
        
        assert prices == {}
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    @patch('src.trading.signal_generation.create_buy_order')
    @patch('src.trading.signal_generation.create_sell_order')
    def test_execute_order_buy(self, mock_create_sell, mock_create_buy, 
                             mock_predictor_class, mock_create_strategy,
                             mock_broker, mock_strategy_config):
        """Test order execution for buy orders."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.return_value = Mock()
        
        mock_order_request = Mock()
        mock_create_buy.return_value = mock_order_request
        
        mock_broker_result = Mock()
        mock_broker_result.get.return_value = True  # success
        mock_broker.place_order.return_value = mock_broker_result
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        order = {'symbol': 'SPY', 'side': 'buy', 'quantity': 10}
        result = generator._execute_order(order)
        
        mock_create_buy.assert_called_once_with(symbol='SPY', qty=10)
        mock_broker.place_order.assert_called_once_with(mock_order_request)
        assert result['success'] is True
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    @patch('src.trading.signal_generation.create_sell_order')
    def test_execute_order_sell(self, mock_create_sell, mock_predictor_class, 
                              mock_create_strategy, mock_broker, mock_strategy_config):
        """Test order execution for sell orders."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.return_value = Mock()
        
        mock_order_request = Mock()
        mock_create_sell.return_value = mock_order_request
        
        mock_broker_result = Mock()
        mock_broker_result.get.return_value = True
        mock_broker.place_order.return_value = mock_broker_result
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        order = {'symbol': 'AAPL', 'side': 'sell', 'quantity': 5}
        result = generator._execute_order(order)
        
        mock_create_sell.assert_called_once_with(symbol='AAPL', qty=5)
        assert result['success'] is True
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_execute_order_invalid_side(self, mock_predictor_class, mock_create_strategy,
                                      mock_broker, mock_strategy_config):
        """Test order execution with invalid side."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.return_value = Mock()
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        order = {'symbol': 'SPY', 'side': 'invalid', 'quantity': 10}
        result = generator._execute_order(order)
        
        assert result['success'] is False
        assert "Unknown side" in result['error']
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_execute_order_exception(self, mock_predictor_class, mock_create_strategy,
                                   mock_broker, mock_strategy_config):
        """Test order execution with exception."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.return_value = Mock()
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        # Mock exception during order creation
        with patch('src.trading.signal_generation.create_buy_order', 
                  side_effect=Exception("Order creation failed")):
            order = {'symbol': 'SPY', 'side': 'buy', 'quantity': 10}
            result = generator._execute_order(order)
        
        assert result['success'] is False
        assert "Order creation failed" in result['error']
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_get_signal_statistics(self, mock_predictor_class, mock_create_strategy,
                                 mock_broker, mock_strategy_config):
        """Test signal statistics retrieval."""
        mock_strategy = Mock()
        mock_strategy.get_status.return_value = {'active': True}
        mock_create_strategy.return_value = mock_strategy
        
        mock_predictor = Mock()
        mock_predictor.get_model_info.return_value = {'model_path': 'test.pth'}
        mock_predictor_class.return_value = mock_predictor
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        stats = generator.get_signal_statistics()
        
        expected_keys = {
            'execution_stats', 'recent_signals_count', 
            'signal_history_count', 'strategy_status', 'model_info'
        }
        assert set(stats.keys()) == expected_keys
        assert 'total_signals' in stats['execution_stats']
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_reset_statistics(self, mock_predictor_class, mock_create_strategy,
                            mock_broker, mock_strategy_config):
        """Test statistics reset."""
        mock_create_strategy.return_value = Mock()
        mock_predictor_class.return_value = Mock()
        
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        # Set some statistics
        generator.execution_stats['total_signals'] = 10
        generator.signal_history = [1, 2, 3]
        generator.recent_signals = {'SPY': 'test'}
        
        generator.reset_statistics()
        
        assert generator.execution_stats['total_signals'] == 0
        assert len(generator.signal_history) == 0
        assert len(generator.recent_signals) == 0


class TestCreateSignalGenerator:
    """Test create_signal_generator factory function."""
    
    @patch('src.trading.signal_generation.SignalGenerator')
    def test_create_signal_generator(self, mock_signal_generator_class):
        """Test signal generator factory function."""
        mock_broker = Mock()
        mock_strategy_config = Mock()
        mock_instance = Mock()
        mock_signal_generator_class.return_value = mock_instance
        
        result = create_signal_generator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        mock_signal_generator_class.assert_called_once_with(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        assert result == mock_instance


class TestSignalGenerationIntegration:
    """Integration tests for signal generation."""
    
    @patch('src.trading.signal_generation.create_strategy')
    @patch('src.trading.signal_generation.LSTMPredictor')
    def test_end_to_end_pipeline(self, mock_predictor_class, mock_create_strategy):
        """Test complete signal generation pipeline."""
        # Setup comprehensive mocks
        mock_broker = Mock()
        mock_broker.place_order.return_value = {'success': True, 'order_id': '12345'}
        
        mock_strategy_config = Mock()
        mock_strategy_config.symbols = ['SPY']
        
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = {
            'SPY': {'action': 'buy', 'confidence': 0.8}
        }
        mock_strategy.process_signals.return_value = [
            {'symbol': 'SPY', 'side': 'buy', 'quantity': 10}
        ]
        mock_create_strategy.return_value = mock_strategy
        
        mock_predictor = Mock()
        mock_predictor.predict_prices.return_value = {
            'SPY': {'5min': {'price': 101.0, 'confidence': 0.8}}
        }
        mock_predictor_class.return_value = mock_predictor
        
        # Create test data
        test_data = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }]).set_index(['symbol', 'timestamp'])
        
        # Create and test signal generator
        generator = SignalGenerator(
            broker=mock_broker,
            strategy_config=mock_strategy_config,
            strategy_type="momentum",
            model_path="test_model.pth"
        )
        
        result = generator.generate_and_execute_signals(test_data)
        
        # Verify complete pipeline execution
        assert result['symbols_processed'] == 1
        assert result['signals_generated'] > 0
        assert result['orders_placed'] > 0
        assert result['orders_successful'] > 0
        assert len(result['errors']) == 0
        
        # Verify all components were called
        mock_predictor.predict_prices.assert_called_once()
        mock_strategy.generate_signals.assert_called_once()
        mock_strategy.process_signals.assert_called_once()
        mock_broker.place_order.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])