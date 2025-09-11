#!/usr/bin/env python3
"""
LSTM Training Pipeline

Configurable training script for multi-horizon stock price prediction LSTM.
Reads configuration from config.py and orchestrates complete training workflow.

Usage:
    python src/models/lstm/training.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.config import TICKERS, TRAINING_START_DATE, TRAINING_END_DATE, VALIDATE_START_DATE, VALIDATE_END_DATE, LSTM_CONFIG
from src.models._utils.data_ingestion import fetch_stock_data
from src.models._utils.feature_engineering import compute_features
from src.models.lstm.model import StockPriceLSTM, StockPriceLoss
from src.utils.logging_config import logger


class LSTMDataset(Dataset):
    """Dataset for LSTM training with multi-horizon targets."""
    
    def __init__(self,
                name: str,
                df: pd.DataFrame, 
                sequence_length: int = 60,
                prediction_horizons: List[int] = [5, 15, 30, 60],
                feature_columns: Optional[List[str]] = None):
        """
        Initialize LSTM dataset.
        
        Args:
            df: DataFrame with MultiIndex (symbol, timestamp)
            sequence_length: Length of input sequences
            prediction_horizons: Minutes ahead to predict
            feature_columns: List of feature column names
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        
        if feature_columns is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
                'vol_15m', 'vol_60m', 'vol_zscore',
                'time_sin', 'time_cos'
            ]
        else:
            self.feature_columns = feature_columns
        
        self.samples = []
        self.targets = []
        
        # Process data by symbol
        for symbol in df.index.get_level_values(0).unique():
            symbol_data = df.loc[symbol].sort_index()
            
            if len(symbol_data) < sequence_length + max(prediction_horizons):
                logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} bars")
                continue
            
            # Create sequences
            for i in range(sequence_length, len(symbol_data) - max(prediction_horizons)):
                # Input sequence
                sequence = symbol_data.iloc[i-sequence_length:i][self.feature_columns].values
                
                # Check for NaN/Inf in sequence
                if np.isnan(sequence).any() or np.isinf(sequence).any():
                    continue
                
                # Multi-horizon targets
                current_price = symbol_data.iloc[i-1]['close']
                horizon_targets = {}
                
                for horizon in prediction_horizons:
                    if i + horizon - 1 < len(symbol_data):
                        future_price = symbol_data.iloc[i + horizon - 1]['close']
                        # Check for valid price
                        if pd.isna(future_price) or np.isinf(future_price) or future_price <= 0:
                            continue
                        horizon_targets[f'price_{horizon}min'] = future_price
                
                # Only add if all horizons available and valid
                if len(horizon_targets) == len(prediction_horizons):
                    self.samples.append(sequence)
                    self.targets.append(horizon_targets)

        logger.info(f"Created dataset '{name}' with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.samples[idx]),
            'targets': {k: torch.FloatTensor([v]) for k, v in self.targets[idx].items()}
        }

class LSTMTrainer:
    """Training pipeline for LSTM model."""
    
    def __init__(self, config: Dict):
        """Initialize trainer with configuration."""
        self.config = config
        
        # Model
        self.model = StockPriceLSTM(**self.config['model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.criterion = StockPriceLoss(self.config['loss']['horizon_weights'])
        
        # Optimizer
        if self.config['optimizer']['type'] == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )
        elif self.config['optimizer']['type'] == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )
        
        # Scheduler
        if self.config['scheduler']['type'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config['scheduler']['patience'],
                factor=self.config['scheduler']['factor']
            )
        elif self.config['scheduler']['type'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_train_loss = float('inf')
        self.epochs_without_improvement = 0
        
        logger.info(f"LSTM Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    

    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation data loaders with separate datasets."""
        logger.info("Preparing training and validation data...")
        
        # Create separate datasets for training and validation
        train_dataset = LSTMDataset(
            "train_dataset",
            train_df,
            sequence_length=self.config['model']['sequence_length'],
            prediction_horizons=self.config['model']['prediction_horizons']
        )
        
        val_dataset = LSTMDataset(
            "val_dataset",
            val_df,
            sequence_length=self.config['model']['sequence_length'],
            prediction_horizons=self.config['model']['prediction_horizons']
        )
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            features = batch['features'].to(self.device)
            targets = {k: v.squeeze().to(self.device) for k, v in batch['targets'].items()}
            
            # Debug: Check for NaN/Inf in inputs
            if torch.isnan(features).any() or torch.isinf(features).any():
                logger.error(f"NaN/Inf detected in features")
                continue
            
            for k, v in targets.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    logger.error(f"NaN/Inf detected in target {k}")
                    continue
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Debug: Check predictions
            for k, v in predictions.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    logger.error(f"NaN/Inf detected in predictions {k}")
                    continue
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
            # Debug: Check loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected: {loss.item()}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = {k: v.squeeze().to(self.device) for k, v in batch['targets'].items()}
                
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training loop."""
        logger.info("Starting LSTM training...")
        
        training_start = datetime.now()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start = datetime.now()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.8f}, Time: {epoch_time:.1f}s"
            )
            
            # Early stopping and model saving
            if val_loss < self.best_train_loss:
                self.best_train_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint('best_lstm_model.pth', epoch, val_loss)
                logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
                
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        training_time = (datetime.now() - training_start).total_seconds()
        
        # Training summary
        summary = {
            'epochs_trained': epoch + 1,
            'best_train_loss': self.best_train_loss,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'training_time_minutes': training_time / 60,
            'config': self.config
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_train_loss:.6f}")
        
        return summary
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        os.makedirs('src/models/lstm/weights', exist_ok=True)
        filepath = os.path.join('src/models/lstm/weights', filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'model_config': self.config
        }, filepath)
    
    def plot_training_curves(self, save_path: str | None):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()


def main():
    """Main training pipeline."""
    logger.info("Starting LSTM training pipeline...")
    
    try:
        # Configuration from config.py
        symbols = TICKERS
        train_start = TRAINING_START_DATE
        train_end = TRAINING_END_DATE
        val_start = VALIDATE_START_DATE
        val_end = VALIDATE_END_DATE

        logger.info(f"Training configuration:")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Training range: {train_start} to {train_end}")
        logger.info(f"  Validation range: {val_start} to {val_end}")
        
        # 1. Data ingestion
        logger.info("Step 1: Data ingestion")
        fetch_stock_data(tickers=symbols, start_date=train_start, end_date=train_end)
        fetch_stock_data(tickers=symbols, start_date=val_start, end_date=val_end)
        
        # 2. Load and combine training data
        logger.info("Step 2: Loading training data")
        train_data = []
        
        for symbol in symbols:
            try:
                file_path = os.path.join("data", f"{symbol}_1min_{train_start}_to_{train_end}.parquet")
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    logger.info(f"Loaded {len(df)} training bars for {symbol}")
                    train_data.append(df)
                else:
                    logger.warning(f"Training data file not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading training data for {symbol}: {e}")
        
        if not train_data:
            raise ValueError("No training data loaded")
        
        train_df = pd.concat(train_data)
        logger.info(f"Combined training dataset: {len(train_df)} total bars")
        
        # 3. Load and combine validation data
        logger.info("Step 3: Loading validation data")
        val_data = []
        
        for symbol in symbols:
            try:
                file_path = os.path.join("data", f"{symbol}_1min_{val_start}_to_{val_end}.parquet")
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    logger.info(f"Loaded {len(df)} validation bars for {symbol}")
                    val_data.append(df)
                else:
                    logger.warning(f"Validation data file not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading validation data for {symbol}: {e}")
        
        if not val_data:
            raise ValueError("No validation data loaded")
        
        val_df = pd.concat(val_data)
        logger.info(f"Combined validation dataset: {len(val_df)} total bars")
        
        # 4. Feature engineering
        logger.info("Step 4: Feature engineering for training data")
        train_featured_df = compute_features(train_df)
        
        if train_featured_df.empty:
            raise ValueError("Training feature computation failed")
        
        logger.info(f"Training features computed: {len(train_featured_df)} bars with {len(train_featured_df.columns)} features")
        
        logger.info("Step 5: Feature engineering for validation data")
        val_featured_df = compute_features(val_df)
        
        if val_featured_df.empty:
            raise ValueError("Validation feature computation failed")
        
        logger.info(f"Validation features computed: {len(val_featured_df)} bars with {len(val_featured_df.columns)} features")
        
        # 6. Initialize trainer
        logger.info("Step 6: Initializing LSTM trainer")
        
        trainer = LSTMTrainer(LSTM_CONFIG)
        
        # 7. Prepare data
        logger.info("Step 7: Preparing data loaders")
        train_loader, val_loader = trainer.prepare_data(train_featured_df, val_featured_df)
        
        # 8. Train model
        logger.info("Step 8: Training LSTM model")
        training_summary = trainer.train(train_loader, val_loader)
        
        # 9. Save final model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f"lstm_model_{timestamp}.pth"
        trainer.save_checkpoint(final_model_path, training_summary['epochs_trained'], training_summary['best_train_loss'])
        
        # 10. Save training summary
        summary_path = f"src/models/lstm/weights/training_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        # 9. Plot training curves
        plot_path = f"src/models/lstm/weights/training_curves_{timestamp}.png"
        trainer.plot_training_curves(plot_path)
        
        logger.info("LSTM training pipeline completed successfully!")
        logger.info(f"Best model: src/models/lstm/weights/{final_model_path}")
        logger.info(f"Training summary: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())