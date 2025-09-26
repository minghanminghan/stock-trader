import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import os
from datetime import datetime

from v2.config import LSTM_MODEL, LSTM_TRAINING
from v2.utils import get_data, preprocess_data, logger
from v2.model import StockPriceLSTM


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock price sequences.

    Creates sliding windows from preprocessed features:
    - Input: (input_length, input_size) sequence
    - Target: (input_size,) next timestep features
    """

    def __init__(self, features: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit_scaler: bool = False):
        """
        Args:
            features: DataFrame with preprocessed features (output from preprocess_data)
            scaler: StandardScaler instance, created if None
            fit_scaler: Whether to fit scaler on this data (True for train, False for val/test)
        """
        self.input_length = LSTM_MODEL['input_length']

        # Handle missing values
        features = features.dropna()

        # Initialize or use provided scaler
        if scaler is None:
            self.scaler = StandardScaler()
            fit_scaler = True
        else:
            self.scaler = scaler

        # Scale features
        if fit_scaler:
            scaled_features = self.scaler.fit_transform(features.values)
        else:
            scaled_features = self.scaler.transform(features.values)

        # Create sequences
        self.sequences, self.targets = self._create_sequences(scaled_features)

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and targets from scaled feature data.

        Args:
            data: Scaled features array (n_timesteps, n_features)

        Returns:
            sequences: Input sequences (n_samples, input_length, n_features)
            targets: Target sequences (n_samples, n_features)
        """
        sequences = []
        targets = []

        # Create sliding windows
        for i in range(self.input_length, len(data)):
            # Input: previous input_length timesteps
            seq = data[i - self.input_length:i]
            # Target: current timestep (what we want to predict)
            target = data[i]

            sequences.append(seq)
            targets.append(target)

        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence: Input tensor (input_length, input_size)
            target: Target tensor (input_size,)
        """
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])


def prepare_training_data(symbols: List[str]) -> Tuple[StockDataset, StockDataset, StockDataset]:
    """
    Prepare training, validation, and test datasets for multiple stocks.
    Uses configuration from LSTM_TRAINING for all parameters.

    Args:
        symbols: List of stock symbols

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    start_date = LSTM_TRAINING['start_date']
    end_date = LSTM_TRAINING['end_date']
    train_ratio = LSTM_TRAINING['train_ratio']
    val_ratio = LSTM_TRAINING['val_ratio']

    logger.info(f"Preparing data for {symbols} from {start_date} to {end_date}")

    # Combine data from all symbols
    all_features = []
    for symbol in symbols:
        logger.info(f"Loading data for {symbol}")
        raw_data = get_data(symbol, start_date, end_date)
        features = preprocess_data(raw_data)
        all_features.append(features)

    # Concatenate all symbol data (chronologically)
    combined_features = pd.concat(all_features, axis=0).sort_index()

    # Temporal split (no shuffling - preserve time order)
    n_samples = len(combined_features)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_features = combined_features.iloc[:train_end]
    val_features = combined_features.iloc[train_end:val_end]
    test_features = combined_features.iloc[val_end:]

    logger.info(f"Data split: train={len(train_features)}, val={len(val_features)}, test={len(test_features)}")

    # Create datasets with shared scaler
    train_dataset = StockDataset(train_features, fit_scaler=True)
    val_dataset = StockDataset(val_features, scaler=train_dataset.scaler)
    test_dataset = StockDataset(test_features, scaler=train_dataset.scaler)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset: StockDataset, val_dataset: StockDataset,
                   test_dataset: StockDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and test datasets.
    Uses batch_size from LSTM_TRAINING configuration.

    Args:
        train_dataset, val_dataset, test_dataset: StockDataset instances

    Returns:
        train_loader, val_loader, test_loader
    """
    batch_size = LSTM_TRAINING['batch_size']

    # No shuffling for time series data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


def create_loss_function():
    """Create combined MSE + MAE loss function using config weights."""
    mse_weight = LSTM_TRAINING['mse_weight']
    mae_weight = LSTM_TRAINING['mae_weight']

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    def combined_loss(predictions, targets):
        return mse_weight * mse_loss(predictions, targets) + mae_weight * mae_loss(predictions, targets)

    return combined_loss


def create_model():
    """Create model instance using config parameters."""
    return StockPriceLSTM(
        input_size=LSTM_MODEL['input_size'],
        output_size=LSTM_MODEL['output_size'],
        hidden_size=LSTM_MODEL['hidden_size'],
        num_layers=LSTM_MODEL['num_layers'],
        dropout=LSTM_MODEL['dropout'],
        input_length=LSTM_MODEL['input_length'],
        output_length=LSTM_MODEL['output_length']
    )


def create_optimizer_and_scheduler(model):
    """Create optimizer and scheduler using config parameters."""
    if LSTM_TRAINING['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LSTM_TRAINING['lr'],
            weight_decay=LSTM_TRAINING['weight_decay']
        )
    else:
        raise ValueError(f"Optimizer {LSTM_TRAINING['optimizer']} not supported")

    scheduler = None
    if LSTM_TRAINING['lr_scheduler']:
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=LSTM_TRAINING['lr_patience'],
            factor=LSTM_TRAINING['lr_factor']
        )

    return optimizer, scheduler


def validate_model(model, val_loader, loss_fn, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_model():
    """Main training function using all config parameters."""
    logger.info("Starting training pipeline...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Prepare data
    logger.info("Preparing training data...")
    train_dataset, val_dataset, test_dataset = prepare_training_data(LSTM_TRAINING['train_symbols'])
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset)

    # Create model, loss, optimizer
    model = create_model().to(device)
    loss_fn = create_loss_function()
    optimizer, scheduler = create_optimizer_and_scheduler(model)

    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Training setup
    epochs = LSTM_TRAINING['epochs']
    gradient_clip_norm = LSTM_TRAINING['gradient_clip_norm']
    early_stopping_patience = LSTM_TRAINING['early_stopping_patience']

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Starting training loop...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Validation phase
        avg_val_loss = validate_model(model, val_loader, loss_fn, device)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Logging
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'weights', 'best_model.pth'))
            logger.info(f"New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Final test evaluation
    logger.info("Training complete. Evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss = validate_model(model, test_loader, loss_fn, device)
    logger.info(f"Final test loss: {test_loss:.6f}")

    return model


if __name__ == '__main__':
    # Start training
    trained_model = train_model()