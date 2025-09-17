#!/usr/bin/env python3
"""
Simplified LSTM Training Pipeline
Simple training script for 60-minute horizon stock price prediction LSTM.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from typing import Dict

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

            # Vectorized target creation
            target_indices = valid_indices[:, None] + np.arange(1, self.prediction_horizon + 1)
            # Clip indices to prevent out-of-bounds
            target_indices = np.clip(target_indices, 0, len(prices_array) - 1)
            targets = prices_array[target_indices]  # Shape: (n_samples, prediction_horizon)

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

        self.best_val_loss = float('inf')
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

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training loop."""
        logger.info("Starting simplified LSTM training...")

        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.scheduler.step(val_loss)

            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_lstm_model.pth')
            else:
                self.epochs_without_improvement += 1

        return {'best_val_loss': self.best_val_loss, 'epochs_trained': epoch + 1}

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        os.makedirs('src/models/lstm/weights', exist_ok=True)
        filepath = os.path.join('src/models/lstm/weights', filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
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
    logger.info("Starting LSTM training pipeline...")

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

        # 5. Train model
        trainer = LSTMTrainer(LSTM_CONFIG)
        summary = trainer.train(train_loader, val_loader)

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {summary['best_val_loss']:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())