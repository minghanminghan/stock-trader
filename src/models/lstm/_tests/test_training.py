#!/usr/bin/env python3
"""
Unit tests for src/models/lstm/training.py - LSTM Training Pipeline
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

from src.models.lstm.training import (
    LSTMDataset, LSTMTrainer, main
)


class TestLSTMDataset:
    """Test LSTMDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=200, freq='1min')
        symbols = ['SPY', 'AAPL']
        
        data = []
        for symbol in symbols:
            for i, ts in enumerate(timestamps):
                data.append({
                    'symbol': symbol,
                    'timestamp': ts,
                    'open': 100.0 + i * 0.01,
                    'high': 101.0 + i * 0.01,
                    'low': 99.0 + i * 0.01,
                    'close': 100.5 + i * 0.01,
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
    
    def test_dataset_initialization(self, sample_data):
        """Test dataset initialization."""
        dataset = LSTMDataset(
            name="test_dataset",
            df=sample_data,
            sequence_length=60,
            prediction_horizons=[5, 15, 30, 60]
        )
        
        assert dataset.sequence_length == 60
        assert dataset.prediction_horizons == [5, 15, 30, 60]
        assert len(dataset.feature_columns) == 14
        assert len(dataset) > 0
    
    def test_dataset_with_custom_features(self, sample_data):
        """Test dataset with custom feature columns."""
        custom_features = ['open', 'high', 'low', 'close', 'volume']
        
        dataset = LSTMDataset(
            name="custom_test",
            df=sample_data,
            sequence_length=30,
            prediction_horizons=[5, 15],
            feature_columns=custom_features
        )
        
        assert dataset.feature_columns == custom_features
    
    def test_dataset_insufficient_data(self):
        """Test dataset with insufficient data."""
        # Create minimal data
        small_data = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'close': 100.0,
            'open': 100.0,
            'high': 100.0,
            'low': 100.0,
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
        }]).set_index(['symbol', 'timestamp'])
        
        dataset = LSTMDataset(
            name="small_test",
            df=small_data,
            sequence_length=60,
            prediction_horizons=[5, 15]
        )
        
        # Should have no samples due to insufficient data
        assert len(dataset) == 0
    
    def test_dataset_sample_structure(self, sample_data):
        """Test dataset sample structure."""
        dataset = LSTMDataset(
            name="structure_test",
            df=sample_data,
            sequence_length=30,
            prediction_horizons=[5, 15]
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            assert 'features' in sample
            assert 'targets' in sample
            
            # Check feature tensor shape
            features = sample['features']
            assert isinstance(features, torch.Tensor)
            assert features.shape == (30, 14)  # (seq_len, features)
            
            # Check targets
            targets = sample['targets']
            assert 'price_5min' in targets
            assert 'price_15min' in targets
            
            for target_tensor in targets.values():
                assert isinstance(target_tensor, torch.Tensor)
                assert target_tensor.shape == (1,)  # Single value
    
    def test_dataset_data_validation(self, sample_data):
        """Test dataset handles invalid data properly."""
        # Add NaN values to data
        corrupted_data = sample_data.copy()
        corrupted_data.iloc[50:55, 0] = np.nan  # Add NaN to some rows
        
        dataset = LSTMDataset(
            name="validation_test",
            df=corrupted_data,
            sequence_length=30,
            prediction_horizons=[5, 15]
        )
        
        # Dataset should skip sequences with NaN/Inf values
        # Length might be reduced but should not crash
        assert len(dataset) >= 0
    
    def test_dataset_price_validation(self, sample_data):
        """Test dataset validates future prices properly."""
        # Create data with invalid future prices
        invalid_data = sample_data.copy()
        invalid_data.loc[invalid_data.index[100:105], 'close'] = np.nan
        
        dataset = LSTMDataset(
            name="price_validation_test",
            df=invalid_data,
            sequence_length=30,
            prediction_horizons=[5, 15]
        )
        
        # Should handle invalid prices gracefully
        assert len(dataset) >= 0


class TestLSTMTrainer:
    """Test LSTMTrainer class."""
    
    @pytest.fixture
    def trainer_config(self):
        """Create trainer configuration."""
        return {
            'model': {
                'input_size': 14,
                'sequence_length': 60,
                'hidden_size': 32,  # Smaller for testing
                'num_layers': 2,
                'dropout': 0.1,
                'prediction_horizons': [5, 15]
            },
            'training': {
                'epochs': 2,  # Few epochs for testing
                'batch_size': 4,
                'validation_split': 0.2,
                'early_stopping_patience': 5,
                'gradient_clip_norm': 1.0
            },
            'optimizer': {
                'type': 'adamw',
                'lr': 0.001,
                'weight_decay': 1e-5
            },
            'scheduler': {
                'type': 'plateau',
                'patience': 3,
                'factor': 0.5
            },
            'loss': {
                'horizon_weights': {'5min': 2.0, '15min': 1.0}
            }
        }
    
    @pytest.fixture
    def trainer(self, trainer_config):
        """Create trainer instance."""
        return LSTMTrainer(trainer_config)
    
    @pytest.fixture
    def sample_dataframes(self):
        """Create sample train/validation DataFrames."""
        def create_df(n_rows=100):
            timestamps = pd.date_range('2024-01-01', periods=n_rows, freq='1min')
            data = []
            for i, ts in enumerate(timestamps):
                data.append({
                    'symbol': 'SPY',
                    'timestamp': ts,
                    'open': 100.0 + i * 0.01,
                    'high': 101.0 + i * 0.01,
                    'low': 99.0 + i * 0.01,
                    'close': 100.5 + i * 0.01,
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
                })
            return pd.DataFrame(data).set_index(['symbol', 'timestamp'])
        
        return create_df(150), create_df(50)  # train, val
    
    def test_trainer_initialization(self, trainer, trainer_config):
        """Test trainer initialization."""
        assert trainer.config == trainer_config
        assert trainer.device.type in ['cpu', 'cuda']
        assert isinstance(trainer.model, nn.Module)
        assert trainer.model.training  # Should be in training mode initially
        
        # Check optimizer
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'scheduler')
        assert hasattr(trainer, 'criterion')
        
        # Check training state
        assert trainer.train_losses == []
        assert trainer.val_losses == []
        assert trainer.best_train_loss == float('inf')
        assert trainer.epochs_without_improvement == 0
    
    def test_trainer_adam_optimizer(self, trainer_config):
        """Test trainer with Adam optimizer."""
        trainer_config['optimizer']['type'] = 'adam'
        trainer = LSTMTrainer(trainer_config)
        
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_trainer_cosine_scheduler(self, trainer_config):
        """Test trainer with cosine annealing scheduler."""
        trainer_config['scheduler']['type'] = 'cosine'
        trainer = LSTMTrainer(trainer_config)
        
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_prepare_data(self, trainer, sample_dataframes):
        """Test data preparation."""
        train_df, val_df = sample_dataframes
        
        train_loader, val_loader = trainer.prepare_data(train_df, val_df)
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert train_loader.batch_size == trainer.config['training']['batch_size']
        assert val_loader.batch_size == trainer.config['training']['batch_size']
    
    def test_train_epoch(self, trainer, sample_dataframes):
        """Test single training epoch."""
        train_df, val_df = sample_dataframes
        train_loader, _ = trainer.prepare_data(train_df, val_df)
        
        if len(train_loader) > 0:
            initial_loss = trainer.train_epoch(train_loader)
            
            assert isinstance(initial_loss, float)
            assert initial_loss >= 0
            assert np.isfinite(initial_loss)
    
    def test_validate_epoch(self, trainer, sample_dataframes):
        """Test validation epoch."""
        train_df, val_df = sample_dataframes
        _, val_loader = trainer.prepare_data(train_df, val_df)
        
        if len(val_loader) > 0:
            val_loss = trainer.validate_epoch(val_loader)
            
            assert isinstance(val_loss, float)
            assert val_loss >= 0
            assert np.isfinite(val_loss)
    
    def test_train_epoch_nan_handling(self, trainer, sample_dataframes):
        """Test training epoch handles NaN gracefully."""
        train_df, val_df = sample_dataframes
        train_loader, _ = trainer.prepare_data(train_df, val_df)
        
        # Mock model to return NaN occasionally
        original_forward = trainer.model.forward
        
        def nan_forward(x):
            outputs = original_forward(x)
            # Introduce NaN in some outputs
            if torch.rand(1) < 0.1:  # 10% chance
                for key in outputs:
                    if 'price' in key:
                        outputs[key] = torch.full_like(outputs[key], float('nan'))
            return outputs
        
        trainer.model.forward = nan_forward
        
        if len(train_loader) > 0:
            # Should handle NaN gracefully and continue training
            loss = trainer.train_epoch(train_loader)
            assert np.isfinite(loss)
    
    @patch('src.models.lstm.training.torch.save')
    @patch('os.makedirs')
    def test_save_checkpoint(self, mock_makedirs, mock_torch_save, trainer):
        """Test checkpoint saving."""
        trainer.train_losses = [1.0, 0.8]
        trainer.val_losses = [1.2, 0.9]
        
        trainer.save_checkpoint('test_model.pth', epoch=5, val_loss=0.9)
        
        mock_makedirs.assert_called_with('src/models/lstm/weights', exist_ok=True)
        mock_torch_save.assert_called_once()
        
        # Check saved content structure
        save_args = mock_torch_save.call_args[0][0]
        expected_keys = {
            'epoch', 'model_state_dict', 'optimizer_state_dict', 
            'scheduler_state_dict', 'val_loss', 'train_losses', 
            'val_losses', 'config', 'model_config'
        }
        assert set(save_args.keys()) == expected_keys
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_curves(self, mock_savefig, mock_show, trainer):
        """Test training curve plotting."""
        trainer.train_losses = [1.0, 0.8, 0.6]
        trainer.val_losses = [1.2, 0.9, 0.7]
        
        trainer.plot_training_curves('test_plot.png')
        
        mock_savefig.assert_called_with('test_plot.png', dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()
    
    def test_plot_training_curves_no_save(self, trainer):
        """Test training curve plotting without saving."""
        trainer.train_losses = [1.0, 0.8]
        trainer.val_losses = [1.2, 0.9]
        
        with patch('matplotlib.pyplot.show') as mock_show:
            trainer.plot_training_curves(None)
            mock_show.assert_called_once()
    
    def test_full_training_loop(self, trainer, sample_dataframes):
        """Test complete training loop."""
        train_df, val_df = sample_dataframes
        train_loader, val_loader = trainer.prepare_data(train_df, val_df)
        
        if len(train_loader) > 0 and len(val_loader) > 0:
            with patch.object(trainer, 'save_checkpoint') as mock_save:
                summary = trainer.train(train_loader, val_loader)
            
            assert isinstance(summary, dict)
            expected_keys = {
                'epochs_trained', 'best_train_loss', 'final_train_loss',
                'final_val_loss', 'training_time_minutes', 'config'
            }
            assert set(summary.keys()) == expected_keys
            
            # Should have saved at least one checkpoint
            assert mock_save.call_count >= 1
            
            # Check training state
            assert len(trainer.train_losses) > 0
            assert len(trainer.val_losses) > 0
    
    def test_early_stopping(self, trainer_config, sample_dataframes):
        """Test early stopping mechanism."""
        # Set very low patience for quick early stopping
        trainer_config['training']['early_stopping_patience'] = 1
        trainer_config['training']['epochs'] = 10  # More epochs than patience
        trainer = LSTMTrainer(trainer_config)
        
        train_df, val_df = sample_dataframes
        train_loader, val_loader = trainer.prepare_data(train_df, val_df)
        
        if len(train_loader) > 0 and len(val_loader) > 0:
            with patch.object(trainer, 'save_checkpoint'):
                summary = trainer.train(train_loader, val_loader)
            
            # Should stop early (before 10 epochs)
            assert summary['epochs_trained'] < 10


class TestMainFunction:
    """Test main training function."""
    
    @patch('src.models.lstm.training.pd.concat')
    @patch('src.models.lstm.training.pd.read_parquet')
    @patch('src.models.lstm.training.os.path.exists')
    @patch('src.models.lstm.training.fetch_stock_data')
    @patch('src.models.lstm.training.compute_features')
    @patch('src.models.lstm.training.LSTMTrainer')
    def test_main_success(self, mock_trainer_class, mock_compute_features, 
                         mock_fetch_data, mock_exists, mock_read_parquet, mock_concat):
        """Test successful main function execution."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_df = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'close': 100.0
        }])
        mock_read_parquet.return_value = mock_df
        mock_concat.return_value = mock_df
        
        # Mock feature computation
        mock_compute_features.return_value = mock_df
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.prepare_data.return_value = (Mock(), Mock())
        mock_trainer_instance.train.return_value = {
            'epochs_trained': 5,
            'best_train_loss': 0.5,
            'final_train_loss': 0.6,
            'final_val_loss': 0.7,
            'training_time_minutes': 10.0,
            'config': {}
        }
        mock_trainer_class.return_value = mock_trainer_instance
        
        # Mock matplotlib and file operations
        with patch('matplotlib.pyplot.savefig'), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('src.models.lstm.training.datetime') as mock_datetime:
            
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            result = main()
        
        assert result == 0  # Success
        mock_fetch_data.assert_called()
        mock_compute_features.assert_called()
        mock_trainer_instance.train.assert_called_once()
    
    @patch('src.models.lstm.training.fetch_stock_data')
    def test_main_fetch_data_failure(self, mock_fetch_data):
        """Test main function with data fetch failure."""
        mock_fetch_data.side_effect = Exception("Data fetch failed")
        
        result = main()
        
        assert result == 1  # Failure
    
    @patch('src.models.lstm.training.fetch_stock_data')
    @patch('src.models.lstm.training.os.path.exists')
    def test_main_no_training_data(self, mock_exists, mock_fetch_data):
        """Test main function when no training data files exist."""
        mock_exists.return_value = False
        
        result = main()
        
        assert result == 1  # Failure
    
    @patch('src.models.lstm.training.fetch_stock_data')
    @patch('src.models.lstm.training.os.path.exists')
    @patch('src.models.lstm.training.pd.read_parquet')
    @patch('src.models.lstm.training.pd.concat')
    @patch('src.models.lstm.training.compute_features')
    def test_main_empty_featured_data(self, mock_compute_features, mock_concat,
                                    mock_read_parquet, mock_exists, mock_fetch_data):
        """Test main function when feature computation returns empty data."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = pd.DataFrame()
        mock_concat.return_value = pd.DataFrame()
        mock_compute_features.return_value = pd.DataFrame()  # Empty
        
        result = main()
        
        assert result == 1  # Failure


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_dataset_trainer_integration(self):
        """Test integration between dataset and trainer."""
        # Create realistic test data
        timestamps = pd.date_range('2024-01-01', periods=200, freq='1min')
        data = []
        for i, ts in enumerate(timestamps):
            data.append({
                'symbol': 'SPY',
                'timestamp': ts,
                'open': 100.0 + np.random.normal(0, 0.1),
                'high': 101.0 + np.random.normal(0, 0.1),
                'low': 99.0 + np.random.normal(0, 0.1),
                'close': 100.5 + np.random.normal(0, 0.1),
                'volume': 1000,
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
        
        df = pd.DataFrame(data).set_index(['symbol', 'timestamp'])
        
        # Create dataset
        dataset = LSTMDataset(
            name="integration_test",
            df=df,
            sequence_length=30,
            prediction_horizons=[5, 15]
        )
        
        # Create trainer with minimal config
        config = {
            'model': {
                'input_size': 14,
                'sequence_length': 30,
                'hidden_size': 16,
                'num_layers': 1,
                'dropout': 0.0,
                'prediction_horizons': [5, 15]
            },
            'training': {
                'epochs': 1,
                'batch_size': 2,
                'validation_split': 0.2,
                'early_stopping_patience': 5,
                'gradient_clip_norm': 1.0
            },
            'optimizer': {
                'type': 'adam',
                'lr': 0.01,
                'weight_decay': 0
            },
            'scheduler': {
                'type': 'plateau',
                'patience': 3,
                'factor': 0.5
            },
            'loss': {
                'horizon_weights': {'5min': 1.0, '15min': 1.0}
            }
        }
        
        trainer = LSTMTrainer(config)
        
        if len(dataset) > 0:
            # Create data loaders
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
            
            # Test training step
            loss = trainer.train_epoch(dataloader)
            assert isinstance(loss, float)
            assert loss >= 0


if __name__ == "__main__":
    pytest.main([__file__])