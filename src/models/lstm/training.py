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
import pickle
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

from src.config import TICKERS, START_DATE, END_DATE
from _utils.data_ingestion import fetch_stock_data
from _utils.feature_engineering import compute_features
from src.models.lstm.model import StockPriceLSTM, StockPriceLoss, create_model
from src.utils.logging_config import logger


class LSTMDataset(Dataset):
    """Dataset for LSTM training with multi-horizon targets."""
    
    def __init__(self, 
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
                
                # Multi-horizon targets
                current_price = symbol_data.iloc[i-1]['close']
                horizon_targets = {}
                
                for horizon in prediction_horizons:
                    if i + horizon - 1 < len(symbol_data):
                        future_price = symbol_data.iloc[i + horizon - 1]['close']
                        horizon_targets[f'price_{horizon}min'] = future_price
                
                # Only add if all horizons available
                if len(horizon_targets) == len(prediction_horizons):
                    self.samples.append(sequence)
                    self.targets.append(horizon_targets)
        
        logger.info(f"Created dataset with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.samples[idx]),
            'targets': {k: torch.FloatTensor([v]) for k, v in self.targets[idx].items()}
        }

default_config = {
    'model': {
        'input_size': 14,
        'sequence_length': 60,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'prediction_horizons': [5, 15, 30, 60]
    },
    'training': {
        'epochs': 100,
        'batch_size': 64,
        'validation_split': 0.2,
        'early_stopping_patience': 20,
        'gradient_clip_norm': 1.0
    },
    'optimizer': {
        'type': 'adamw',
        'lr': 0.001,
        'weight_decay': 1e-5
    },
    'scheduler': {
        'type': 'plateau',
        'patience': 10,
        'factor': 0.5
    },
    'loss': {
        'horizon_weights': {
            '5min': 2.0,
            '15min': 1.5,
            '30min': 1.0,
            '60min': 0.8
        }
    }
}

class LSTMTrainer:
    """Training pipeline for LSTM model."""
    
    def __init__(self, config: Dict = default_config):
        """Initialize trainer with configuration."""
        self.config = config
        
        # Model
        self.model = create_model(self.config['model'])
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
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        logger.info(f"LSTM Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    

    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation data loaders."""
        logger.info("Preparing training data...")
        
        # Create dataset
        dataset = LSTMDataset(
            df,
            sequence_length=self.config['model']['sequence_length'],
            prediction_horizons=self.config['model']['prediction_horizons']
        )
        
        # Train/validation split
        val_size = int(len(dataset) * self.config['training']['validation_split'])
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
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
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
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
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
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
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
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
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'training_time_minutes': training_time / 60,
            'config': self.config
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
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
            'model_config': self.model.get_model_config()
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
        start_date = START_DATE  # Training start
        end_date = END_DATE      # Training end
        
        logger.info(f"Training configuration:")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        
        # 1. Data ingestion
        logger.info("Step 1: Data ingestion")
        fetch_stock_data(tickers=symbols, start_date=start_date, end_date=end_date)
        
        # 2. Load and combine data
        logger.info("Step 2: Loading and combining data")
        all_data = []
        
        for symbol in symbols:
            try:
                file_path = os.path.join("data", f"{symbol}_1min_{start_date}_to_{end_date}.parquet")
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                    all_data.append(df)
                else:
                    logger.warning(f"Data file not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        if not all_data:
            raise ValueError("No data loaded for training")
        
        combined_df = pd.concat(all_data)
        logger.info(f"Combined dataset: {len(combined_df)} total bars")
        
        # 3. Feature engineering
        logger.info("Step 3: Feature engineering")
        featured_df = compute_features(combined_df)
        
        if featured_df.empty:
            raise ValueError("Feature computation failed")
        
        logger.info(f"Features computed: {len(featured_df)} bars with {len(featured_df.columns)} features")
        
        # 4. Initialize trainer
        logger.info("Step 4: Initializing LSTM trainer")
        
        # Custom config (optional)
        custom_config = {
            'model': {
                'input_size': 14,  # Adjust based on actual features
                'sequence_length': 60,
                'hidden_size': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'prediction_horizons': [5, 15, 30, 60]
            },
            'training': {
                'epochs': 100,
                'batch_size': 64,
                'validation_split': 0.2,
                'early_stopping_patience': 20,
                'gradient_clip_norm': 1.0
            }
        }
        
        trainer = LSTMTrainer(custom_config)
        
        # 5. Prepare data
        logger.info("Step 5: Preparing training data")
        train_loader, val_loader = trainer.prepare_data(featured_df)
        
        # 6. Train model
        logger.info("Step 6: Training LSTM model")
        training_summary = trainer.train(train_loader, val_loader)
        
        # 7. Save final model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f"lstm_model_{timestamp}.pth"
        trainer.save_checkpoint(final_model_path, training_summary['epochs_trained'], training_summary['best_val_loss'])
        
        # 8. Save training summary
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