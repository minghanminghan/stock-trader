#!/usr/bin/env python3
"""
Simplified LSTM Training Pipeline
Simple training script for 60-minute horizon stock price prediction LSTM.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.config import TICKERS, TRAINING_START_DATE, TRAINING_END_DATE, VALIDATE_START_DATE, VALIDATE_END_DATE, LSTM_CONFIG
from src.models._utils.data_ingestion import fetch_stock_data
from src.models._utils.feature_engineering import compute_features
from src.models.lstm.model import StockPriceLSTM
from src.utils.logging_config import logger


class LSTMDataset(Dataset):
    """In-memory dataset for LSTM training."""

    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, prediction_horizon: int = 60):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
            'vol_15m', 'vol_60m', 'vol_zscore',
            'time_sin', 'time_cos'
        ]

        self.sequences = []
        self.targets = []

        self._create_sequences(data)

    def _create_sequences(self, data: pd.DataFrame):
        """Create sequences from data using vectorized operations."""
        all_sequences = []
        all_targets = []

        for symbol in data.index.get_level_values(0).unique():
            symbol_data = data.loc[symbol].sort_index()

            if len(symbol_data) < self.sequence_length + self.prediction_horizon:
                continue

            # Extract feature and price arrays for vectorized operations
            features_array = symbol_data[self.feature_columns].values.astype(np.float32)
            prices_array = symbol_data['close'].values.astype(np.float32)

            # Calculate valid sequence indices
            max_start_idx = len(symbol_data) - self.prediction_horizon
            valid_indices = np.arange(self.sequence_length, max_start_idx)

            if len(valid_indices) == 0:
                continue

            # Vectorized sequence creation using advanced indexing
            seq_indices = valid_indices[:, None] - np.arange(self.sequence_length, 0, -1)
            sequences = features_array[seq_indices]  # Shape: (n_samples, seq_len, n_features)

            # Vectorized target creation - using percentage changes instead of absolute prices
            target_indices = valid_indices[:, None] + np.arange(1, self.prediction_horizon + 1)
            # Clip indices to prevent out-of-bounds
            target_indices = np.clip(target_indices, 0, len(prices_array) - 1)
            future_prices = prices_array[target_indices]  # Shape: (n_samples, prediction_horizon)

            # Get current prices for each sequence (price at the end of each sequence)
            current_prices = prices_array[valid_indices][:, None]  # Shape: (n_samples, 1)

            # Calculate percentage changes: (future_price - current_price) / current_price
            # Multiply by 100 to get percentage scale (easier to learn than 0.01 scale)
            targets = ((future_prices - current_prices) / current_prices) * 100  # Shape: (n_samples, prediction_horizon)

            # Batch validation - check for NaN values
            sequence_valid = ~np.isnan(sequences).any(axis=(1, 2))
            target_valid = ~np.isnan(targets).any(axis=1)
            valid_mask = sequence_valid & target_valid

            # Filter valid samples
            if valid_mask.sum() > 0:
                all_sequences.append(sequences[valid_mask])
                all_targets.append(targets[valid_mask])

        # Concatenate all valid sequences and targets
        if all_sequences:
            self.sequences = np.concatenate(all_sequences, axis=0)
            self.targets = np.concatenate(all_targets, axis=0)
        else:
            self.sequences = np.empty((0, self.sequence_length, len(self.feature_columns)), dtype=np.float32)
            self.targets = np.empty((0, self.prediction_horizon), dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.sequences[idx]),
            'target': torch.FloatTensor(self.targets[idx])
        }


class LSTMTrainer:
    """Simplified training pipeline for LSTM model"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.model = StockPriceLSTM(
            input_size=self.config['model']['input_size'],
            sequence_length=self.config['model']['sequence_length'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            prediction_horizon=self.config['model']['prediction_horizon']
        ).to(self.device)

        # MSE loss
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['optimizer']['lr'],
            weight_decay=self.config['optimizer']['weight_decay']
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['scheduler']['patience'],
            factor=self.config['scheduler']['factor']
        )

        self.best_train_loss = float('inf')
        self.epochs_without_improvement = 0

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        valid_batches = 0

        for batch in train_loader:
            features = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(features)
            loss = self.criterion(predictions['price'], targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

        return total_loss / max(valid_batches, 1)

    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = batch['target'].to(self.device)

                predictions = self.model(features)
                loss = self.criterion(predictions['price'], targets)

                total_loss += loss.item()
                valid_batches += 1

        return total_loss / max(valid_batches, 1)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model from checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Next epoch to start training from
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load training state
            self.best_train_loss = checkpoint.get('best_train_loss', float('inf'))
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            start_epoch = checkpoint.get('epoch', 0)

            logger.info(f"Loaded checkpoint from epoch {start_epoch}")
            logger.info(f"Best validation loss: {self.best_train_loss:.4f}")
            logger.info(f"Epochs without improvement: {self.epochs_without_improvement}")

            return start_epoch + 1  # Return next epoch to start from

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch")
            return 0

    def train(self, train_loader: DataLoader, val_loader: DataLoader, resume_from_checkpoint: Optional[str] = None) -> Dict:
        """Complete training loop with optional checkpoint resumption."""

        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)

        logger.info(f"Starting LSTM training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['training']['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)




            self.scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping and checkpointing
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_log_lstm_model.pth', epoch, train_loss, val_loss)
                logger.info(f"New best model saved with training loss: {train_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Save regular checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, train_loss, val_loss)

            # Early stopping check
            patience = self.config['training'].get('early_stopping_patience', 15)
            if self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        return {'best_train_loss': self.best_train_loss, 'epochs_trained': epoch + 1}

    def save_checkpoint(self, filename: str, epoch: int, train_loss: float, val_loss: float):
        """Save comprehensive model checkpoint."""
        os.makedirs('src/models/lstm/weights', exist_ok=True)
        filepath = os.path.join('src/models/lstm/weights', filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_train_loss': self.best_train_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
        }, filepath)


def load_data(symbols, start_date: str, end_date: str) -> pd.DataFrame:
    """Load and process data for all symbols."""
    dataframes = []

    for symbol in symbols:
        try:
            file_path = os.path.join("data", f"{symbol}_1min_{start_date}_to_{end_date}.parquet")
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue

            df = pd.read_parquet(file_path)
            featured_df = compute_features(df)

            if not featured_df.empty:
                dataframes.append(featured_df)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    if not dataframes:
        raise ValueError("No data loaded")

    return pd.concat(dataframes, axis=0)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LSTM Stock Price Prediction Training')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train (overrides config)')
    args = parser.parse_args()

    logger.info("Starting LSTM training pipeline...")

    # Log training mode
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
    else:
        logger.info("Starting fresh training")

    try:
        # 1. Data ingestion
        fetch_stock_data(tickers=TICKERS, start_date=TRAINING_START_DATE, end_date=TRAINING_END_DATE)
        fetch_stock_data(tickers=TICKERS, start_date=VALIDATE_START_DATE, end_date=VALIDATE_END_DATE)

        # 2. Load and process data
        logger.info("Loading training data...")
        train_data = load_data(TICKERS, TRAINING_START_DATE, TRAINING_END_DATE)

        logger.info("Loading validation data...")
        val_data = load_data(TICKERS, VALIDATE_START_DATE, VALIDATE_END_DATE)

        # 3. Create datasets
        train_dataset = LSTMDataset(
            train_data,
            sequence_length=LSTM_CONFIG['model']['sequence_length'],
            prediction_horizon=LSTM_CONFIG['model']['prediction_horizon']
        )

        val_dataset = LSTMDataset(
            val_data,
            sequence_length=LSTM_CONFIG['model']['sequence_length'],
            prediction_horizon=LSTM_CONFIG['model']['prediction_horizon']
        )

        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # 4. Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 5. Override config if epochs specified
        config = LSTM_CONFIG.copy()
        if args.epochs:
            config['training']['epochs'] = args.epochs
            logger.info(f"Overriding epochs to {args.epochs}")

        # 6. Train model with optional checkpoint resumption
        trainer = LSTMTrainer(config)
        summary = trainer.train(train_loader, val_loader, resume_from_checkpoint=args.resume)

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {summary['best_train_loss']:.4f}")
        logger.info(f"Total epochs trained: {summary['epochs_trained']}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())