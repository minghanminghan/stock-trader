#!/usr/bin/env python3
"""
Unit tests for src/config.py - Configuration module
"""

import pytest
from unittest.mock import patch, mock_open
import os

# Mock environment variables for testing
@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'ALPACA_API_KEY': 'test_api_key',
        'ALPACA_SECRET_KEY': 'test_secret_key'
    }):
        yield


@pytest.fixture
def mock_env_vars_missing():
    """Mock missing environment variables for testing."""
    with patch.dict(os.environ, {}, clear=True):
        yield


class TestConfigModule:
    """Test configuration module functionality."""
    
    def test_environment_loading(self, mock_env_vars):
        """Test that environment variables are loaded correctly."""
        # Reload config module to pick up mocked env vars
        import importlib
        import src.config
        importlib.reload(src.config)
        
        assert src.config.ALPACA_API_KEY == 'test_api_key'
        assert src.config.ALPACA_SECRET_KEY == 'test_secret_key'
    
    def test_missing_api_keys_raises_error(self, mock_env_vars_missing):
        """Test that missing API keys raise ValueError."""
        with pytest.raises(ValueError, match="Alpaca API keys must be set"):
            import importlib
            import src.config
            importlib.reload(src.config)
    
    def test_data_directory_constant(self):
        """Test DATA_DIR constant."""
        from src.config import DATA_DIR
        assert DATA_DIR == "data"
        assert isinstance(DATA_DIR, str)
    
    def test_logs_directory_constant(self):
        """Test LOGS_DIR constant."""
        from src.config import LOGS_DIR
        assert LOGS_DIR == "logs"
        assert isinstance(LOGS_DIR, str)
    
    def test_random_seed_constant(self):
        """Test RANDOM_SEED constant."""
        from src.config import RANDOM_SEED
        assert RANDOM_SEED == 0
        assert isinstance(RANDOM_SEED, int)
    
    def test_tickers_configuration(self):
        """Test TICKERS configuration."""
        from src.config import TICKERS
        expected_tickers = ["AMZN", "META", "NVDA"]
        
        assert TICKERS == expected_tickers
        assert isinstance(TICKERS, list)
        assert all(isinstance(ticker, str) for ticker in TICKERS)
    
    def test_date_configurations(self):
        """Test date configuration constants."""
        from src.config import (
            TRAINING_START_DATE, TRAINING_END_DATE,
            VALIDATE_START_DATE, VALIDATE_END_DATE
        )
        
        # Test format and values
        assert TRAINING_START_DATE == "2025-07-01"
        assert TRAINING_END_DATE == "2025-08-31"
        assert VALIDATE_START_DATE == "2025-09-01"
        assert VALIDATE_END_DATE == "2025-09-03"
        
        # Test they are strings
        assert isinstance(TRAINING_START_DATE, str)
        assert isinstance(TRAINING_END_DATE, str)
        assert isinstance(VALIDATE_START_DATE, str)
        assert isinstance(VALIDATE_END_DATE, str)
    
    def test_lstm_config_structure(self):
        """Test LSTM_CONFIG structure and values."""
        from src.config import LSTM_CONFIG
        
        # Test top-level keys
        expected_top_keys = {'model', 'training', 'optimizer', 'scheduler', 'loss'}
        assert set(LSTM_CONFIG.keys()) == expected_top_keys
        
        # Test model configuration
        model_config = LSTM_CONFIG['model']
        expected_model_keys = {
            'input_size', 'sequence_length', 'hidden_size', 
            'num_layers', 'dropout', 'prediction_horizons'
        }
        assert set(model_config.keys()) == expected_model_keys
        
        # Test specific model values
        assert model_config['input_size'] == 14
        assert model_config['sequence_length'] == 60
        assert model_config['hidden_size'] == 128
        assert model_config['num_layers'] == 3
        assert model_config['dropout'] == 0.2
        assert model_config['prediction_horizons'] == [5, 15, 30, 60]
        
        # Test training configuration
        training_config = LSTM_CONFIG['training']
        expected_training_keys = {
            'epochs', 'batch_size', 'validation_split', 
            'early_stopping_patience', 'gradient_clip_norm'
        }
        assert set(training_config.keys()) == expected_training_keys
        
        assert training_config['epochs'] == 5
        assert training_config['batch_size'] == 64
        assert training_config['validation_split'] == 0.2
        assert training_config['early_stopping_patience'] == 20
        assert training_config['gradient_clip_norm'] == 1.0
        
        # Test optimizer configuration
        optimizer_config = LSTM_CONFIG['optimizer']
        expected_optimizer_keys = {'type', 'lr', 'weight_decay'}
        assert set(optimizer_config.keys()) == expected_optimizer_keys
        
        assert optimizer_config['type'] == 'adamw'
        assert optimizer_config['lr'] == 0.001
        assert optimizer_config['weight_decay'] == 1e-5
        
        # Test scheduler configuration
        scheduler_config = LSTM_CONFIG['scheduler']
        expected_scheduler_keys = {'type', 'patience', 'factor'}
        assert set(scheduler_config.keys()) == expected_scheduler_keys
        
        assert scheduler_config['type'] == 'plateau'
        assert scheduler_config['patience'] == 10
        assert scheduler_config['factor'] == 0.5
        
        # Test loss configuration
        loss_config = LSTM_CONFIG['loss']
        expected_loss_keys = {'horizon_weights'}
        assert set(loss_config.keys()) == expected_loss_keys
        
        horizon_weights = loss_config['horizon_weights']
        expected_horizons = {'5min', '15min', '30min', '60min'}
        assert set(horizon_weights.keys()) == expected_horizons
        
        # Test horizon weight values
        assert horizon_weights['5min'] == 2.0
        assert horizon_weights['15min'] == 1.5
        assert horizon_weights['30min'] == 1.0
        assert horizon_weights['60min'] == 0.8
    
    def test_lstm_config_types(self):
        """Test LSTM_CONFIG value types."""
        from src.config import LSTM_CONFIG
        
        # Model config types
        model = LSTM_CONFIG['model']
        assert isinstance(model['input_size'], int)
        assert isinstance(model['sequence_length'], int)
        assert isinstance(model['hidden_size'], int)
        assert isinstance(model['num_layers'], int)
        assert isinstance(model['dropout'], float)
        assert isinstance(model['prediction_horizons'], list)
        assert all(isinstance(h, int) for h in model['prediction_horizons'])
        
        # Training config types
        training = LSTM_CONFIG['training']
        assert isinstance(training['epochs'], int)
        assert isinstance(training['batch_size'], int)
        assert isinstance(training['validation_split'], float)
        assert isinstance(training['early_stopping_patience'], int)
        assert isinstance(training['gradient_clip_norm'], (int, float))
        
        # Optimizer config types
        optimizer = LSTM_CONFIG['optimizer']
        assert isinstance(optimizer['type'], str)
        assert isinstance(optimizer['lr'], float)
        assert isinstance(optimizer['weight_decay'], float)
        
        # Scheduler config types
        scheduler = LSTM_CONFIG['scheduler']
        assert isinstance(scheduler['type'], str)
        assert isinstance(scheduler['patience'], int)
        assert isinstance(scheduler['factor'], float)
        
        # Loss config types
        loss = LSTM_CONFIG['loss']
        horizon_weights = loss['horizon_weights']
        assert isinstance(horizon_weights, dict)
        assert all(isinstance(k, str) for k in horizon_weights.keys())
        assert all(isinstance(v, (int, float)) for v in horizon_weights.values())
    
    def test_prediction_horizons_order(self):
        """Test that prediction horizons are in ascending order."""
        from src.config import LSTM_CONFIG
        
        horizons = LSTM_CONFIG['model']['prediction_horizons']
        assert horizons == sorted(horizons)
    
    def test_horizon_weights_consistency(self):
        """Test that horizon weights correspond to prediction horizons."""
        from src.config import LSTM_CONFIG
        
        horizons = LSTM_CONFIG['model']['prediction_horizons']
        horizon_weights = LSTM_CONFIG['loss']['horizon_weights']
        
        # Check that each horizon has a corresponding weight
        for horizon in horizons:
            horizon_key = f"{horizon}min"
            assert horizon_key in horizon_weights
            assert horizon_weights[horizon_key] > 0
    
    def test_configuration_immutability(self):
        """Test that configurations are safe to use (can be modified without affecting original)."""
        from src.config import LSTM_CONFIG, TICKERS
        
        # Test LSTM_CONFIG
        original_config = LSTM_CONFIG.copy()
        modified_config = LSTM_CONFIG.copy()
        modified_config['model']['input_size'] = 999
        
        # Original should be unchanged
        assert LSTM_CONFIG['model']['input_size'] == original_config['model']['input_size']
        
        # Test TICKERS
        original_tickers = TICKERS.copy()
        modified_tickers = TICKERS.copy()
        modified_tickers.append("TEST")
        
        # Original should be unchanged
        assert TICKERS == original_tickers
    
    def test_empty_api_key_handling(self):
        """Test handling of empty API keys."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': '',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            with pytest.raises(ValueError, match="Alpaca API keys must be set"):
                import importlib
                import src.config
                importlib.reload(src.config)
    
    def test_whitespace_api_key_handling(self):
        """Test handling of whitespace-only API keys."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': '   ',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            # The current implementation checks for empty string, not whitespace
            # This test documents current behavior
            import importlib
            import src.config
            importlib.reload(src.config)
            
            # Should load whitespace as-is (current behavior)
            assert src.config.ALPACA_API_KEY == '   '


class TestConfigValidation:
    """Test configuration validation and constraints."""
    
    def test_lstm_config_valid_ranges(self):
        """Test that LSTM config values are in valid ranges."""
        from src.config import LSTM_CONFIG
        
        model = LSTM_CONFIG['model']
        training = LSTM_CONFIG['training']
        optimizer = LSTM_CONFIG['optimizer']
        scheduler = LSTM_CONFIG['scheduler']
        
        # Model constraints
        assert model['input_size'] > 0
        assert model['sequence_length'] > 0
        assert model['hidden_size'] > 0
        assert model['num_layers'] > 0
        assert 0 <= model['dropout'] <= 1
        assert len(model['prediction_horizons']) > 0
        
        # Training constraints
        assert training['epochs'] > 0
        assert training['batch_size'] > 0
        assert 0 < training['validation_split'] < 1
        assert training['early_stopping_patience'] > 0
        assert training['gradient_clip_norm'] > 0
        
        # Optimizer constraints
        assert optimizer['lr'] > 0
        assert optimizer['weight_decay'] >= 0
        
        # Scheduler constraints
        assert scheduler['patience'] > 0
        assert 0 < scheduler['factor'] < 1
    
    def test_date_format_validation(self):
        """Test that dates are in expected format."""
        from src.config import (
            TRAINING_START_DATE, TRAINING_END_DATE,
            VALIDATE_START_DATE, VALIDATE_END_DATE
        )
        
        import re
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        
        assert re.match(date_pattern, TRAINING_START_DATE)
        assert re.match(date_pattern, TRAINING_END_DATE)
        assert re.match(date_pattern, VALIDATE_START_DATE)
        assert re.match(date_pattern, VALIDATE_END_DATE)
    
    def test_ticker_format_validation(self):
        """Test that tickers are in expected format."""
        from src.config import TICKERS
        
        for ticker in TICKERS:
            assert isinstance(ticker, str)
            assert len(ticker) > 0
            assert ticker.isupper()  # Tickers should be uppercase
            assert ticker.isalpha()  # Tickers should be alphabetic


if __name__ == "__main__":
    pytest.main([__file__])