#!/usr/bin/env python3
"""
Unit tests for src/models/lstm/predictor.py - LSTM Model Predictor
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from src.models.lstm.predictor import LSTMPredictor


class TestLSTMPredictor:
    """Test LSTMPredictor class."""
    
    @pytest.fixture
    def mock_model_path(self):
        """Mock model path for testing."""
        return "test_model.pth"
    
    @pytest.fixture
    def mock_checkpoint(self):
        """Mock model checkpoint data."""
        return {
            'model_state_dict': {
                'input_norm.weight': torch.randn(14),  # 14 features
                'lstm.weight_ih_l0': torch.randn(512, 14),
                'prediction_heads.5min.0.weight': torch.randn(64, 128)
            },
            'model_config': {
                'model': {
                    'input_size': 14,
                    'sequence_length': 60,
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.2,
                    'prediction_horizons': [5, 15, 30, 60]
                }
            }
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=100, freq='1min')
        symbols = ['SPY', 'AAPL']
        
        data = []
        for symbol in symbols:
            for i, ts in enumerate(timestamps):
                data.append({
                    'symbol': symbol,
                    'timestamp': ts,
                    'open': 100.0 + i * 0.1,
                    'high': 101.0 + i * 0.1,
                    'low': 99.0 + i * 0.1,
                    'close': 100.5 + i * 0.1,
                    'volume': 1000 + i,
                    'return_1m': np.random.normal(0, 0.01),
                    'mom_5m': np.random.normal(0, 0.02),
                    'mom_15m': np.random.normal(0, 0.03),
                    'mom_60m': np.random.normal(0, 0.05),
                    'vol_15m': np.random.uniform(0.01, 0.05),
                    'vol_60m': np.random.uniform(0.01, 0.03),
                    'vol_zscore': np.random.normal(0, 1),
                    'time_sin': np.sin(2 * np.pi * i / 100),
                    'time_cos': np.cos(2 * np.pi * i / 100)
                })
        
        df = pd.DataFrame(data)
        return df.set_index(['symbol', 'timestamp'])
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_initialization_with_checkpoint(self, mock_lstm_class, mock_torch_load, 
                                          mock_model_path, mock_checkpoint):
        """Test predictor initialization with valid checkpoint."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_lstm_class.return_value = mock_model_instance
        
        predictor = LSTMPredictor(mock_model_path)
        
        assert predictor.model_path == mock_model_path
        assert predictor.device.type in ['cpu', 'cuda']
        mock_torch_load.assert_called_once()
        mock_lstm_class.assert_called_once()
        mock_model_instance.load_state_dict.assert_called_once()
        mock_model_instance.eval.assert_called_once()
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_initialization_infer_config(self, mock_lstm_class, mock_torch_load, mock_model_path):
        """Test predictor initialization when inferring config from state_dict."""
        # Checkpoint without model_config
        checkpoint_no_config = {
            'model_state_dict': {
                'input_norm.weight': torch.randn(14),  # 14 features
                'lstm.weight_ih_l0': torch.randn(512, 14),
            }
        }
        mock_torch_load.return_value = checkpoint_no_config
        mock_model_instance = Mock()
        mock_lstm_class.return_value = mock_model_instance
        
        with patch('src.models.lstm.predictor.LSTM_CONFIG', {
            'model': {
                'sequence_length': 60,
                'hidden_size': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'prediction_horizons': [5, 15, 30, 60]
            }
        }):
            predictor = LSTMPredictor(mock_model_path)
        
        # Should infer input_size and use default config
        expected_config = {
            'input_size': 14,  # Inferred from state_dict
            'sequence_length': 60,
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'prediction_horizons': [5, 15, 30, 60]
        }
        
        mock_lstm_class.assert_called_with(**expected_config)
    
    @patch('torch.load')
    def test_initialization_failure(self, mock_torch_load, mock_model_path):
        """Test predictor initialization failure."""
        mock_torch_load.side_effect = Exception("Failed to load model")
        
        with pytest.raises(Exception):
            LSTMPredictor(mock_model_path)
    
    @patch('torch.load')
    def test_initialization_missing_state_dict_key(self, mock_torch_load, mock_model_path):
        """Test initialization when state_dict doesn't have expected keys."""
        checkpoint_invalid = {
            'model_state_dict': {}  # Empty state dict
        }
        mock_torch_load.return_value = checkpoint_invalid
        
        with pytest.raises(ValueError, match="Cannot infer model architecture"):
            LSTMPredictor(mock_model_path)
    
    @patch('src.models.lstm.predictor.Path.glob')
    @patch('src.models.lstm.predictor.Path.exists')
    def test_find_latest_model_success(self, mock_exists, mock_glob):
        """Test finding latest model file."""
        mock_exists.return_value = True
        
        # Mock model files with different timestamps
        mock_files = [
            Mock(stat=Mock(return_value=Mock(st_mtime=1000))),
            Mock(stat=Mock(return_value=Mock(st_mtime=2000))),  # Latest
            Mock(stat=Mock(return_value=Mock(st_mtime=1500)))
        ]
        mock_glob.return_value = mock_files
        
        with patch('torch.load'), patch('src.models.lstm.predictor.StockPriceLSTM'):
            predictor = LSTMPredictor(None)  # None triggers latest model search
        
        # Should call _find_latest_model
        mock_exists.assert_called()
        mock_glob.assert_called()
    
    @patch('src.models.lstm.predictor.Path.exists')
    def test_find_latest_model_no_directory(self, mock_exists):
        """Test finding latest model when directory doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="No model weights directory found"):
            predictor = LSTMPredictor(None)
    
    @patch('src.models.lstm.predictor.Path.glob')
    @patch('src.models.lstm.predictor.Path.exists')
    def test_find_latest_model_no_files(self, mock_exists, mock_glob):
        """Test finding latest model when no model files exist."""
        mock_exists.return_value = True
        mock_glob.return_value = []  # No model files
        
        with pytest.raises(FileNotFoundError, match="No trained LSTM models found"):
            predictor = LSTMPredictor(None)
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_predict_prices_success(self, mock_lstm_class, mock_torch_load, 
                                  mock_checkpoint, sample_market_data):
        """Test successful price prediction."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_model_instance.prediction_horizons = [5, 15, 30, 60]
        
        # Mock model predictions
        mock_predictions = {
            'price_5min': torch.tensor([101.0, 102.0]),
            'price_15min': torch.tensor([101.5, 102.5]),
            'price_30min': torch.tensor([102.0, 103.0]),
            'price_60min': torch.tensor([102.5, 103.5]),
            'confidence_5min': torch.tensor([0.8, 0.9]),
            'confidence_15min': torch.tensor([0.7, 0.8]),
            'confidence_30min': torch.tensor([0.6, 0.7]),
            'confidence_60min': torch.tensor([0.5, 0.6]),
            'variance_5min': torch.tensor([1.0, 1.1]),
            'variance_15min': torch.tensor([1.2, 1.3]),
            'variance_30min': torch.tensor([1.4, 1.5]),
            'variance_60min': torch.tensor([1.6, 1.7])
        }
        mock_model_instance.return_value = mock_predictions
        mock_lstm_class.return_value = mock_model_instance
        
        with patch('src.models.lstm.predictor.compute_features', return_value=sample_market_data):
            predictor = LSTMPredictor("test_model.pth")
            predictions = predictor.predict_prices(sample_market_data)
        
        # Should have predictions for both symbols
        assert 'SPY' in predictions
        assert 'AAPL' in predictions
        
        # Check prediction structure
        for symbol in ['SPY', 'AAPL']:
            symbol_preds = predictions[symbol]
            assert '5min' in symbol_preds
            assert '15min' in symbol_preds
            assert '30min' in symbol_preds
            assert '60min' in symbol_preds
            
            for horizon_key in symbol_preds:
                pred = symbol_preds[horizon_key]
                assert 'price' in pred
                assert 'confidence' in pred
                assert 'variance' in pred
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_predict_prices_insufficient_data(self, mock_lstm_class, mock_torch_load, mock_checkpoint):
        """Test prediction with insufficient data."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_model_instance.prediction_horizons = [5, 15, 30, 60]
        mock_lstm_class.return_value = mock_model_instance
        
        # Create small dataset (insufficient data)
        small_data = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'close': 100.0
        }]).set_index(['symbol', 'timestamp'])
        
        with patch('src.models.lstm.predictor.compute_features', return_value=small_data):
            predictor = LSTMPredictor("test_model.pth")
            predictions = predictor.predict_prices(small_data)
        
        # Should return fallback predictions
        assert 'SPY' in predictions
        spy_pred = predictions['SPY']
        assert spy_pred['5min']['confidence'] == 0.1  # Low confidence fallback
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_predict_prices_model_error(self, mock_lstm_class, mock_torch_load, 
                                      mock_checkpoint, sample_market_data):
        """Test prediction when model raises error."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_model_instance.prediction_horizons = [5, 15, 30, 60]
        mock_model_instance.side_effect = Exception("Model error")
        mock_lstm_class.return_value = mock_model_instance
        
        with patch('src.models.lstm.predictor.compute_features', return_value=sample_market_data):
            predictor = LSTMPredictor("test_model.pth")
            predictions = predictor.predict_prices(sample_market_data)
        
        # Should return empty predictions on error
        assert predictions == {}
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_prepare_sequence_success(self, mock_lstm_class, mock_torch_load, 
                                    mock_checkpoint, sample_market_data):
        """Test successful sequence preparation."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_lstm_class.return_value = mock_model_instance
        
        predictor = LSTMPredictor("test_model.pth")
        
        # Get data for one symbol
        spy_data = sample_market_data.loc['SPY']
        
        sequence = predictor._prepare_sequence(spy_data)
        
        assert sequence is not None
        assert isinstance(sequence, torch.Tensor)
        assert sequence.dim() == 3  # (batch=1, seq_len, features)
        assert sequence.shape[0] == 1  # Batch size
        assert sequence.shape[2] == 14  # Number of features
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_prepare_sequence_insufficient_data(self, mock_lstm_class, mock_torch_load, mock_checkpoint):
        """Test sequence preparation with insufficient data."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_lstm_class.return_value = mock_model_instance
        
        predictor = LSTMPredictor("test_model.pth")
        
        # Create insufficient data
        small_data = pd.DataFrame([{
            'open': 100.0,
            'close': 100.0
        }])
        
        sequence = predictor._prepare_sequence(small_data)
        
        assert sequence is None
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_prepare_sequence_nan_values(self, mock_lstm_class, mock_torch_load, mock_checkpoint):
        """Test sequence preparation with NaN values."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_lstm_class.return_value = mock_model_instance
        
        predictor = LSTMPredictor("test_model.pth")
        
        # Create data with NaN values
        timestamps = pd.date_range('2024-01-01', periods=70, freq='1min')
        data_with_nan = pd.DataFrame({
            'open': [100.0] * 69 + [np.nan],  # NaN in last value
            'high': [101.0] * 70,
            'low': [99.0] * 70,
            'close': [100.0] * 70,
            'volume': [1000] * 70,
            'return_1m': [0.01] * 70,
            'mom_5m': [0.02] * 70,
            'mom_15m': [0.03] * 70,
            'mom_60m': [0.05] * 70,
            'vol_15m': [0.02] * 70,
            'vol_60m': [0.01] * 70,
            'vol_zscore': [0.0] * 70,
            'time_sin': [0.5] * 70,
            'time_cos': [0.5] * 70
        }, index=timestamps)
        
        sequence = predictor._prepare_sequence(data_with_nan)
        
        assert sequence is None  # Should return None due to NaN
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_get_fallback_prediction(self, mock_lstm_class, mock_torch_load, mock_checkpoint):
        """Test fallback prediction generation."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_model_instance.prediction_horizons = [5, 15, 30, 60]
        mock_lstm_class.return_value = mock_model_instance
        
        predictor = LSTMPredictor("test_model.pth")
        fallback = predictor._get_fallback_prediction()
        
        assert isinstance(fallback, dict)
        assert '5min' in fallback
        assert '15min' in fallback
        assert '30min' in fallback
        assert '60min' in fallback
        
        for horizon_key in fallback:
            pred = fallback[horizon_key]
            assert 'price' in pred
            assert 'confidence' in pred
            assert 'variance' in pred
            assert pred['confidence'] == 0.1  # Low confidence
            assert pred['variance'] == 25.0    # High uncertainty
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_get_model_info(self, mock_lstm_class, mock_torch_load, mock_checkpoint):
        """Test model information retrieval."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_model_instance.prediction_horizons = [5, 15, 30, 60]
        mock_model_instance.parameters.return_value = [torch.randn(100), torch.randn(50)]
        mock_lstm_class.return_value = mock_model_instance
        
        predictor = LSTMPredictor("test_model.pth")
        info = predictor.get_model_info()
        
        assert 'model_path' in info
        assert 'device' in info
        assert 'config' in info
        assert 'prediction_horizons' in info
        assert 'input_features' in info
        assert 'model_parameters' in info
        
        assert info['model_path'] == "test_model.pth"
        assert info['prediction_horizons'] == [5, 15, 30, 60]
        assert len(info['input_features']) == 14  # Expected feature count
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_feature_columns_completeness(self, mock_lstm_class, mock_torch_load, mock_checkpoint):
        """Test that feature columns match expected LSTM input features."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_lstm_class.return_value = mock_model_instance
        
        predictor = LSTMPredictor("test_model.pth")
        
        expected_features = [
            'open', 'high', 'low', 'close', 'volume',
            'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
            'vol_15m', 'vol_60m', 'vol_zscore',
            'time_sin', 'time_cos'
        ]
        
        assert predictor.feature_columns == expected_features
        assert len(predictor.feature_columns) == 14
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_cuda_availability_handling(self, mock_lstm_class, mock_torch_load, mock_checkpoint):
        """Test CUDA device handling."""
        mock_torch_load.return_value = mock_checkpoint
        mock_model_instance = Mock()
        mock_lstm_class.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=True):
            predictor = LSTMPredictor("test_model.pth")
            # Device should be set based on CUDA availability
            assert predictor.device.type in ['cpu', 'cuda']
        
        with patch('torch.cuda.is_available', return_value=False):
            predictor = LSTMPredictor("test_model.pth")
            assert predictor.device.type == 'cpu'


class TestLSTMPredictorIntegration:
    """Integration tests for LSTMPredictor."""
    
    @patch('torch.load')
    @patch('src.models.lstm.predictor.StockPriceLSTM')
    def test_end_to_end_prediction_pipeline(self, mock_lstm_class, mock_torch_load):
        """Test complete prediction pipeline."""
        # Setup mocks
        mock_checkpoint = {
            'model_state_dict': {'input_norm.weight': torch.randn(14)},
            'model_config': {
                'model': {
                    'input_size': 14,
                    'sequence_length': 60,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'prediction_horizons': [5, 15]
                }
            }
        }
        mock_torch_load.return_value = mock_checkpoint
        
        mock_model_instance = Mock()
        mock_model_instance.prediction_horizons = [5, 15]
        mock_predictions = {
            'price_5min': torch.tensor([101.0]),
            'price_15min': torch.tensor([101.5]),
            'confidence_5min': torch.tensor([0.8]),
            'confidence_15min': torch.tensor([0.7]),
            'variance_5min': torch.tensor([1.0]),
            'variance_15min': torch.tensor([1.2])
        }
        mock_model_instance.return_value = mock_predictions
        mock_lstm_class.return_value = mock_model_instance
        
        # Create test data
        test_data = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000,
            'return_1m': 0.01,
            'mom_5m': 0.02,
            'mom_15m': 0.03,
            'mom_60m': 0.05,
            'vol_15m': 0.02,
            'vol_60m': 0.01,
            'vol_zscore': 0.0,
            'time_sin': 0.5,
            'time_cos': 0.5
        } for _ in range(70)]).set_index(['symbol', 'timestamp'])
        
        # Test prediction
        predictor = LSTMPredictor("test_model.pth")
        predictions = predictor.predict_prices(test_data)
        
        # Verify results
        assert 'SPY' in predictions
        spy_pred = predictions['SPY']
        assert '5min' in spy_pred
        assert '15min' in spy_pred
        
        # Check prediction values
        assert spy_pred['5min']['price'] == 101.0
        assert spy_pred['15min']['price'] == 101.5
        assert spy_pred['5min']['confidence'] == 0.8
        assert spy_pred['15min']['confidence'] == 0.7


if __name__ == "__main__":
    pytest.main([__file__])