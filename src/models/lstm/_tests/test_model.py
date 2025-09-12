#!/usr/bin/env python3
"""
Unit tests for src/models/lstm/model.py - LSTM Model for Multi-Horizon Prediction
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch

from src.models.lstm.model import (
    StockPriceLSTM, StockPriceLoss, count_parameters
)


class TestStockPriceLSTM:
    """Test StockPriceLSTM model class."""
    
    @pytest.fixture
    def model_config(self):
        """Standard model configuration for testing."""
        return {
            'input_size': 14,
            'sequence_length': 60,
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'prediction_horizons': [5, 15, 30, 60]
        }
    
    @pytest.fixture
    def model(self, model_config):
        """Create test model instance."""
        return StockPriceLSTM(**model_config)
    
    def test_model_initialization(self, model, model_config):
        """Test model initialization with correct parameters."""
        assert model.input_size == model_config['input_size']
        assert model.sequence_length == model_config['sequence_length']
        assert model.hidden_size == model_config['hidden_size']
        assert model.num_layers == model_config['num_layers']
        assert model.dropout == model_config['dropout']
        assert model.prediction_horizons == model_config['prediction_horizons']
        assert model.num_horizons == len(model_config['prediction_horizons'])
    
    def test_model_components(self, model):
        """Test that all model components are properly initialized."""
        # Input normalization
        assert isinstance(model.input_norm, nn.BatchNorm1d)
        assert model.input_norm.num_features == model.input_size
        
        # LSTM backbone
        assert isinstance(model.lstm, nn.LSTM)
        assert model.lstm.input_size == model.input_size
        assert model.lstm.hidden_size == model.hidden_size
        assert model.lstm.num_layers == model.num_layers
        assert model.lstm.batch_first is True
        
        # Attention mechanism
        assert isinstance(model.attention, nn.MultiheadAttention)
        assert model.attention.embed_dim == model.hidden_size
        
        # Feature extractor
        assert isinstance(model.feature_extractor, nn.Sequential)
        
        # Prediction heads
        assert len(model.prediction_heads) == model.num_horizons
        assert len(model.confidence_heads) == model.num_horizons
        
        for horizon in model.prediction_horizons:
            horizon_key = f'{horizon}min'
            assert horizon_key in model.prediction_heads
            assert horizon_key in model.confidence_heads
    
    def test_forward_pass_shape(self, model):
        """Test forward pass produces correct output shapes."""
        batch_size = 8
        seq_len = model.sequence_length
        input_size = model.input_size
        
        # Create test input
        x = torch.randn(batch_size, seq_len, input_size)
        
        # Forward pass
        outputs = model(x)
        
        # Check output structure
        expected_keys = []
        for horizon in model.prediction_horizons:
            horizon_key = f'{horizon}min'
            expected_keys.extend([
                f'price_{horizon_key}',
                f'confidence_{horizon_key}',
                f'variance_{horizon_key}'
            ])
        
        assert set(outputs.keys()) == set(expected_keys)
        
        # Check output shapes
        for horizon in model.prediction_horizons:
            horizon_key = f'{horizon}min'
            assert outputs[f'price_{horizon_key}'].shape == (batch_size,)
            assert outputs[f'confidence_{horizon_key}'].shape == (batch_size,)
            assert outputs[f'variance_{horizon_key}'].shape == (batch_size,)
    
    def test_forward_pass_values(self, model):
        """Test forward pass produces reasonable values."""
        batch_size = 4
        x = torch.randn(batch_size, model.sequence_length, model.input_size)
        
        with torch.no_grad():
            outputs = model(x)
        
        # Check that outputs are finite
        for key, tensor in outputs.items():
            assert torch.isfinite(tensor).all(), f"Non-finite values in {key}"
        
        # Check confidence values are in [0, 1]
        for horizon in model.prediction_horizons:
            horizon_key = f'{horizon}min'
            confidence = outputs[f'confidence_{horizon_key}']
            assert (confidence >= 0).all() and (confidence <= 1).all()
        
        # Check variance values are positive
        for horizon in model.prediction_horizons:
            horizon_key = f'{horizon}min'
            variance = outputs[f'variance_{horizon_key}']
            assert (variance > 0).all()
    
    def test_minimal_config(self):
        """Test model with minimal configuration."""
        config = {
            'input_size': 5,
            'sequence_length': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.0,
            'prediction_horizons': [5]
        }
        
        model = StockPriceLSTM(**config)
        x = torch.randn(2, 10, 5)
        
        outputs = model(x)
        
        assert 'price_5min' in outputs
        assert 'confidence_5min' in outputs
        assert 'variance_5min' in outputs
    
    def test_single_layer_lstm(self):
        """Test model with single LSTM layer (no dropout in LSTM)."""
        config = {
            'input_size': 10,
            'sequence_length': 20,
            'hidden_size': 64,
            'num_layers': 1,  # Single layer should have no dropout
            'dropout': 0.3,
            'prediction_horizons': [5, 10]
        }
        
        model = StockPriceLSTM(**config)
        
        # LSTM should not have dropout for single layer
        assert model.lstm.dropout == 0
    
    def test_multiple_horizons(self):
        """Test model with multiple prediction horizons."""
        horizons = [1, 5, 15, 30, 60, 120]
        config = {
            'input_size': 8,
            'sequence_length': 30,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'prediction_horizons': horizons
        }
        
        model = StockPriceLSTM(**config)
        x = torch.randn(3, 30, 8)
        
        outputs = model(x)
        
        # Should have outputs for all horizons
        for horizon in horizons:
            horizon_key = f'{horizon}min'
            assert f'price_{horizon_key}' in outputs
            assert f'confidence_{horizon_key}' in outputs
            assert f'variance_{horizon_key}' in outputs
    
    def test_gradient_flow(self, model):
        """Test that gradients flow properly through the model."""
        batch_size = 4
        x = torch.randn(batch_size, model.sequence_length, model.input_size, requires_grad=True)
        
        outputs = model(x)
        
        # Create a simple loss by summing all price predictions
        total_loss = sum(outputs[key] for key in outputs.keys() if 'price_' in key).sum()
        
        total_loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_model_device_transfer(self, model):
        """Test model can be moved to different devices."""
        # Test CPU (always available)
        model_cpu = model.to('cpu')
        x_cpu = torch.randn(2, model.sequence_length, model.input_size)
        
        outputs_cpu = model_cpu(x_cpu)
        assert all(tensor.device.type == 'cpu' for tensor in outputs_cpu.values())
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            x_cuda = torch.randn(2, model.sequence_length, model.input_size).cuda()
            
            outputs_cuda = model_cuda(x_cuda)
            assert all(tensor.device.type == 'cuda' for tensor in outputs_cuda.values())
    
    def test_batch_size_flexibility(self, model):
        """Test model works with different batch sizes."""
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, model.sequence_length, model.input_size)
            outputs = model(x)
            
            # All outputs should have correct batch size
            for tensor in outputs.values():
                assert tensor.shape[0] == batch_size
    
    def test_deterministic_output(self, model):
        """Test model produces deterministic output in eval mode."""
        model.eval()
        
        x = torch.randn(4, model.sequence_length, model.input_size)
        
        with torch.no_grad():
            outputs1 = model(x)
            outputs2 = model(x)
        
        # Outputs should be identical in eval mode
        for key in outputs1.keys():
            assert torch.allclose(outputs1[key], outputs2[key], atol=1e-6)
    
    def test_input_validation(self, model):
        """Test model handles incorrect input shapes appropriately."""
        # Wrong sequence length
        x_wrong_seq = torch.randn(4, 30, model.input_size)  # Wrong sequence length
        
        # Should still work, just with different sequence length
        outputs = model(x_wrong_seq)
        assert len(outputs) > 0
        
        # Wrong input size should raise error
        x_wrong_input = torch.randn(4, model.sequence_length, 5)  # Wrong input size
        
        with pytest.raises(RuntimeError):
            model(x_wrong_input)


class TestStockPriceLoss:
    """Test StockPriceLoss class."""
    
    @pytest.fixture
    def horizon_weights(self):
        """Standard horizon weights for testing."""
        return {
            '5min': 2.0,
            '15min': 1.5,
            '30min': 1.0,
            '60min': 0.8
        }
    
    @pytest.fixture
    def loss_fn(self, horizon_weights):
        """Create test loss function."""
        return StockPriceLoss(horizon_weights)
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions for testing."""
        batch_size = 4
        return {
            'price_5min': torch.randn(batch_size),
            'price_15min': torch.randn(batch_size),
            'price_30min': torch.randn(batch_size),
            'price_60min': torch.randn(batch_size),
            'variance_5min': torch.exp(torch.randn(batch_size)),  # Positive variance
            'variance_15min': torch.exp(torch.randn(batch_size)),
            'variance_30min': torch.exp(torch.randn(batch_size)),
            'variance_60min': torch.exp(torch.randn(batch_size)),
        }
    
    @pytest.fixture
    def sample_targets(self):
        """Sample targets for testing."""
        batch_size = 4
        return {
            'price_5min': torch.randn(batch_size),
            'price_15min': torch.randn(batch_size),
            'price_30min': torch.randn(batch_size),
            'price_60min': torch.randn(batch_size),
        }
    
    def test_loss_initialization(self, loss_fn, horizon_weights):
        """Test loss function initialization."""
        assert loss_fn.horizon_weights == horizon_weights
    
    def test_default_horizon_weights(self):
        """Test loss function with default horizon weights."""
        loss_fn = StockPriceLoss()
        
        expected_defaults = {
            '5min': 2.0,
            '15min': 1.5,
            '30min': 1.0,
            '60min': 0.8
        }
        
        assert loss_fn.horizon_weights == expected_defaults
    
    def test_loss_computation(self, loss_fn, sample_predictions, sample_targets):
        """Test basic loss computation."""
        loss = loss_fn(sample_predictions, sample_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
        assert torch.isfinite(loss)
    
    def test_loss_without_variance(self, loss_fn, sample_targets):
        """Test loss computation without variance predictions."""
        predictions_no_var = {
            'price_5min': torch.randn(4),
            'price_15min': torch.randn(4),
            'price_30min': torch.randn(4),
            'price_60min': torch.randn(4),
            # No variance predictions
        }
        
        loss = loss_fn(predictions_no_var, sample_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
    
    def test_loss_gradient_flow(self, loss_fn, sample_targets):
        """Test that loss allows gradient flow."""
        predictions = {
            'price_5min': torch.randn(4, requires_grad=True),
            'price_15min': torch.randn(4, requires_grad=True),
            'variance_5min': torch.exp(torch.randn(4, requires_grad=True)),
            'variance_15min': torch.exp(torch.randn(4, requires_grad=True)),
        }
        
        targets = {
            'price_5min': sample_targets['price_5min'],
            'price_15min': sample_targets['price_15min'],
        }
        
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        # Check gradients exist
        assert predictions['price_5min'].grad is not None
        assert predictions['price_15min'].grad is not None
        assert predictions['variance_5min'].grad is not None
        assert predictions['variance_15min'].grad is not None
    
    def test_loss_with_mismatched_keys(self, loss_fn):
        """Test loss computation with mismatched prediction/target keys."""
        predictions = {
            'price_5min': torch.randn(4),
            'price_15min': torch.randn(4),
        }
        
        targets = {
            'price_30min': torch.randn(4),  # Different key
            'price_60min': torch.randn(4),
        }
        
        loss = loss_fn(predictions, targets)
        
        # Should return zero loss when no matching keys
        assert loss.item() == 0.0
    
    def test_loss_horizon_weighting(self, horizon_weights):
        """Test that horizon weights are properly applied."""
        loss_fn = StockPriceLoss(horizon_weights)
        
        batch_size = 4
        
        # Create predictions and targets with different errors for each horizon
        predictions = {}
        targets = {}
        
        for horizon_key, weight in horizon_weights.items():
            # Create larger error for higher weighted horizons
            if weight > 1.5:  # High weight horizons
                predictions[f'price_{horizon_key}'] = torch.ones(batch_size) * 10
                targets[f'price_{horizon_key}'] = torch.zeros(batch_size)  # Large error
            else:  # Lower weight horizons
                predictions[f'price_{horizon_key}'] = torch.ones(batch_size) * 2
                targets[f'price_{horizon_key}'] = torch.zeros(batch_size)  # Smaller error
        
        loss = loss_fn(predictions, targets)
        
        # Loss should be weighted more by high-weight horizons
        assert loss.item() > 0
    
    def test_loss_with_extreme_variance(self, loss_fn, sample_targets):
        """Test loss computation with extreme variance values."""
        predictions = {
            'price_5min': torch.randn(4),
            'variance_5min': torch.tensor([1e-8, 1e8, 1e-6, 1e6])  # Extreme values
        }
        
        targets = {
            'price_5min': sample_targets['price_5min']
        }
        
        loss = loss_fn(predictions, targets)
        
        # Loss should still be finite despite extreme variance
        assert torch.isfinite(loss)
    
    def test_loss_nan_handling(self, loss_fn, sample_targets):
        """Test loss handles NaN values in variance gracefully."""
        predictions = {
            'price_5min': torch.randn(4),
            'variance_5min': torch.tensor([1.0, float('nan'), 1.0, 1.0])
        }
        
        targets = {
            'price_5min': sample_targets['price_5min']
        }
        
        # Should handle NaN gracefully (implementation sets uncertainty_loss to 0)
        loss = loss_fn(predictions, targets)
        assert torch.isfinite(loss)
    
    def test_empty_predictions_targets(self, loss_fn):
        """Test loss with empty predictions and targets."""
        loss = loss_fn({}, {})
        
        # Should return zero loss
        assert loss.item() == 0.0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_count_parameters(self):
        """Test parameter counting function."""
        # Simple model for testing
        model = nn.Sequential(
            nn.Linear(10, 5),  # 10*5 + 5 = 55 parameters
            nn.Linear(5, 1)    # 5*1 + 1 = 6 parameters
        )
        
        total_params = count_parameters(model)
        assert total_params == 61  # 55 + 6
    
    def test_count_parameters_with_frozen(self):
        """Test parameter counting with frozen parameters."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 1)
        )
        
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False
        
        trainable_params = count_parameters(model)
        assert trainable_params == 6  # Only second layer parameters
    
    def test_count_parameters_lstm_model(self):
        """Test parameter counting on LSTM model."""
        config = {
            'input_size': 5,
            'sequence_length': 10,
            'hidden_size': 16,
            'num_layers': 1,
            'dropout': 0.0,
            'prediction_horizons': [5]
        }
        
        model = StockPriceLSTM(**config)
        param_count = count_parameters(model)
        
        assert param_count > 0
        assert isinstance(param_count, int)


class TestModelIntegration:
    """Integration tests for the complete model."""
    
    def test_training_step_simulation(self):
        """Test a complete training step simulation."""
        config = {
            'input_size': 8,
            'sequence_length': 20,
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.1,
            'prediction_horizons': [5, 15]
        }
        
        model = StockPriceLSTM(**config)
        loss_fn = StockPriceLoss({'5min': 2.0, '15min': 1.0})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Simulate training batch
        batch_size = 8
        x = torch.randn(batch_size, config['sequence_length'], config['input_size'])
        targets = {
            'price_5min': torch.randn(batch_size),
            'price_15min': torch.randn(batch_size),
        }
        
        # Forward pass
        model.train()
        predictions = model(x)
        loss = loss_fn(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
        
        # Optimizer step
        optimizer.step()
        
        assert loss.item() >= 0
    
    def test_inference_mode(self):
        """Test model in inference mode."""
        config = {
            'input_size': 6,
            'sequence_length': 15,
            'hidden_size': 24,
            'num_layers': 1,
            'dropout': 0.2,
            'prediction_horizons': [5, 30]
        }
        
        model = StockPriceLSTM(**config)
        model.eval()
        
        x = torch.randn(1, config['sequence_length'], config['input_size'])
        
        with torch.no_grad():
            predictions = model(x)
        
        # Should have all expected outputs
        expected_keys = ['price_5min', 'price_30min', 'confidence_5min', 
                        'confidence_30min', 'variance_5min', 'variance_30min']
        assert set(predictions.keys()) == set(expected_keys)
        
        # All outputs should be scalars for batch_size=1
        for tensor in predictions.values():
            assert tensor.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__])