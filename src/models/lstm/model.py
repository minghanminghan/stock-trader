#!/usr/bin/env python3
"""
LSTM Model for Multi-Horizon Stock Price Prediction with Confidence Estimation

Predicts stock prices at 5, 15, 30, and 60-minute horizons with uncertainty quantification.
Designed for Alpaca Markets minute-level OHLCV data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class StockPriceLSTM(nn.Module):
    """
    Multi-horizon LSTM for stock price prediction with confidence estimation.
    
    Architecture:
    - Input: Sequence of OHLCV + technical indicators
    - LSTM layers with dropout for temporal modeling
    - Multi-head output for different prediction horizons
    - Uncertainty estimation via dropout at inference time
    """
    
    def __init__(self,
        input_size: int,            # OHLCV + technical indicators
        sequence_length: int,       # 60 minutes of history
        hidden_size: int,           # LSTM hidden units
        num_layers: int,            # LSTM layers
        dropout: float,             # Dropout rate
        prediction_horizons: list,  # Minutes ahead
        **kwargs
        ):
        """
        Initialize LSTM model for multi-horizon price prediction.
        
        Args:
            input_size: Number of input features per timestep
            sequence_length: Number of historical timesteps to use
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
            prediction_horizons: List of minutes ahead to predict
        """
        super(StockPriceLSTM, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_horizons = prediction_horizons
        self.num_horizons = len(prediction_horizons)
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism for focusing on important timesteps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict()
        self.confidence_heads = nn.ModuleDict()
        
        for horizon in prediction_horizons:
            # Price prediction head
            self.prediction_heads[f'{horizon}min'] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1)  # Single price prediction
            )
            
            # Confidence estimation head (predicts log-variance)
            self.confidence_heads[f'{horizon}min'] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1)  # Log-variance for uncertainty
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Dictionary containing predictions and confidence for each horizon
        """
        batch_size, seq_len, input_size = x.shape
        
        # Input normalization (reshape for BatchNorm1d)
        x_norm = x.permute(0, 2, 1)  # (batch, features, time)
        x_norm = self.input_norm(x_norm)
        x_norm = x_norm.permute(0, 2, 1)  # Back to (batch, time, features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_norm)
        
        # Self-attention over sequence
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Combine LSTM output with attention
        combined = lstm_out + attended_out
        
        # Use last timestep for prediction
        last_hidden = combined[:, -1, :]  # (batch_size, hidden_size)
        
        # Feature extraction
        features = self.feature_extractor(last_hidden)
        
        # Multi-horizon predictions
        outputs = {}
        
        for horizon in self.prediction_horizons:
            horizon_key = f'{horizon}min'
            
            # Price prediction
            price_pred = self.prediction_heads[horizon_key](features)
            outputs[f'price_{horizon_key}'] = price_pred.squeeze(-1)
            
            # Confidence prediction (log-variance)
            log_var = self.confidence_heads[horizon_key](features)
            
            # Clamp log_var to prevent explosion
            log_var = torch.clamp(log_var, min=-10, max=10)
            
            # Convert log-variance to variance with stability
            variance = torch.exp(log_var)
            variance = torch.clamp(variance, min=1e-6, max=1e6)
            
            confidence = torch.sigmoid(-variance)  # Higher variance = lower confidence
            outputs[f'confidence_{horizon_key}'] = confidence.squeeze(-1)
            outputs[f'variance_{horizon_key}'] = variance.squeeze(-1)
        
        return outputs

class StockPriceLoss(nn.Module):
    """
    Custom loss function for multi-horizon price prediction with uncertainty.
    
    Combines:
    - MSE loss for price predictions
    - Negative log-likelihood for uncertainty estimation
    - Horizon-weighted loss (shorter horizons more important)
    """
    
    def __init__(self, horizon_weights: Optional[Dict[str, float]] = None):
        """
        Initialize loss function.
        
        Args:
            horizon_weights: Weight for each prediction horizon
        """
        super(StockPriceLoss, self).__init__()
        
        # Default weights (shorter horizons more important)
        if horizon_weights is None:
            self.horizon_weights = {
                '5min': 2.0,
                '15min': 1.5,
                '30min': 1.0,
                '60min': 0.8
            }
        else:
            self.horizon_weights = horizon_weights
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute multi-horizon loss with uncertainty.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Combined loss tensor
        """
        total_loss = 0.0
        num_horizons = 0
        
        for horizon_key in self.horizon_weights.keys():
            if f'price_{horizon_key}' in predictions and f'price_{horizon_key}' in targets:
                # Price prediction loss (MSE)
                price_pred = predictions[f'price_{horizon_key}']
                price_target = targets[f'price_{horizon_key}']
                mse_loss = F.mse_loss(price_pred, price_target)
                
                # Uncertainty loss (negative log-likelihood)
                if f'variance_{horizon_key}' in predictions:
                    variance = predictions[f'variance_{horizon_key}']
                    # Clamp variance to avoid NaN/Inf
                    variance = torch.clamp(variance, min=1e-6, max=1e6)
                    # Negative log-likelihood for Gaussian
                    uncertainty_loss = 0.5 * (
                        torch.log(variance) + 
                        (price_pred - price_target).pow(2) / variance
                    ).mean()
                    
                    # Additional check for NaN
                    if torch.isnan(uncertainty_loss) or torch.isinf(uncertainty_loss):
                        uncertainty_loss = 0.0
                else:
                    uncertainty_loss = 0.0
                
                # Weighted combination
                horizon_loss = mse_loss + 0.1 * uncertainty_loss
                weight = self.horizon_weights[horizon_key]
                total_loss += weight * horizon_loss
                num_horizons += 1
        
        return total_loss / max(num_horizons, 1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    example_config = {
        "input_size": 14,
        "sequence_length": 60,
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "prediction_horizons": [5, 15, 30, 60]
    }
    model = StockPriceLSTM(**example_config)
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    print(f"Model structure:\n{model}")