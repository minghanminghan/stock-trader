#!/usr/bin/env python3
"""
LSTM Model Predictor for Live Trading

Handles loading trained LSTM models and generating real-time predictions
with confidence scores for multi-horizon stock price forecasting.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import glob
from pathlib import Path

from src.models.lstm.model import StockPriceLSTM
from src.models._utils.feature_engineering import compute_features
from src.utils.logging_config import logger


class LSTMPredictor:
    """
    LSTM model predictor for real-time trading signals.
    
    Handles model loading, data preprocessing, and multi-horizon price predictions
    with confidence estimation for trading decision making.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize LSTM predictor.
        
        Args:
            model_path: Path to saved model (.pth file). If None, loads latest model
            device: Device to run model on ('cpu', 'cuda'). If None, auto-detects
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = None
        self.model_config = {}
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
            'vol_15m', 'vol_60m', 'vol_zscore',
            'time_sin', 'time_cos'
        ]
        
        # Load model
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = self._find_latest_model()
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get model configuration
            self.model_config = checkpoint.get('model_config', {})
            
            # If no config saved, infer from model state_dict
            if not self.model_config:
                logger.warning("No model config found in checkpoint, inferring from state_dict")
                state_dict = checkpoint['model_state_dict']
                
                # Infer input_size from input normalization layer
                if 'input_norm.weight' in state_dict:
                    input_size = state_dict['input_norm.weight'].shape[0]
                    logger.info(f"Inferred input_size: {input_size}")
                    
                    # Use config from src.config with inferred input_size
                    from src.config import LSTM_CONFIG
                    self.model_config = LSTM_CONFIG['model'].copy()
                    self.model_config['input_size'] = input_size
                else:
                    raise ValueError("Cannot infer model architecture from checkpoint")
            
            # Create model with saved/inferred config
            self.model = StockPriceLSTM(**self.model_config['model'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded LSTM model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
            logger.info(f"Model prediction horizons: {self.model.prediction_horizons}")
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise
        
        logger.info(f"LSTM Predictor initialized with model: {self.model_path}")
        logger.info(f"Running on device: {self.device}")
    
    def _find_latest_model(self) -> str:
        """Find the latest trained model file."""
        weights_dir = Path("src/models/lstm/weights")
        
        if not weights_dir.exists():
            raise FileNotFoundError("No model weights directory found")
        
        # Look for timestamped models
        model_files = list(weights_dir.glob("lstm_model_*.pth"))
        if not model_files:
            raise FileNotFoundError("No trained LSTM models found")
        
        # Return most recent
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return str(latest_model)
    
    def predict_prices(self, market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Generate multi-horizon price predictions with native model confidence using batch processing.
        
        Args:
            market_data: DataFrame with OHLCV data and MultiIndex (symbol, timestamp)
            
        Returns:
            Dict mapping symbol -> horizon -> {price, confidence, variance}
        """
        predictions = {}
        
        try:
            # Ensure we have features
            if not all(col in market_data.columns for col in self.feature_columns):
                featured_data = compute_features(market_data)
            else:
                featured_data = market_data
            
            # Collect sequences for batch processing
            sequences = []
            symbols = []
            fallback_symbols = []
            
            for symbol in featured_data.index.get_level_values(0).unique():
                symbol_data = featured_data.loc[symbol].sort_index()
                
                if len(symbol_data) < self.model_config.get('sequence_length', 60):
                    logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} bars")
                    fallback_symbols.append(symbol)
                    continue
                
                # Prepare input sequence
                sequence = self._prepare_sequence(symbol_data)
                
                if sequence is None:
                    fallback_symbols.append(symbol)
                    continue
                
                sequences.append(sequence)
                symbols.append(symbol)
            
            # Add fallback predictions for failed symbols
            for symbol in fallback_symbols:
                predictions[symbol] = self._get_fallback_prediction()
            
            # Batch predict if we have valid sequences
            if sequences:
                logger.info(f"Batch predicting for {len(symbols)} symbols: {symbols}")
                
                # Stack sequences into batch tensor
                batch_sequences = torch.cat(sequences, dim=0)  # (batch_size, seq_len, features)
                logger.info(f"Batch tensor shape: {batch_sequences.shape}")
                
                # Single forward pass for all symbols
                with torch.no_grad():
                    batch_outputs = self.model(batch_sequences)
                    logger.info(f"Batch outputs keys: {list(batch_outputs.keys())}")
                
                # Split batch results by symbol
                for i, symbol in enumerate(symbols):
                    symbol_predictions = {}
                    
                    for horizon in self.model.prediction_horizons:
                        horizon_key = f'{horizon}min'
                        price_key = f'price_{horizon_key}'
                        
                        if price_key not in batch_outputs:
                            logger.error(f"Missing price key {price_key} in batch outputs")
                            continue
                        
                        # Extract prediction for this symbol from batch
                        price_pred = batch_outputs[price_key][i].cpu().item()
                        
                        result_dict = {'price': price_pred}
                        
                        # Extract confidence and variance if available
                        confidence_key = f'confidence_{horizon_key}'
                        variance_key = f'variance_{horizon_key}'
                        
                        if confidence_key in batch_outputs and variance_key in batch_outputs:
                            confidence = batch_outputs[confidence_key][i].cpu().item()
                            variance = batch_outputs[variance_key][i].cpu().item()
                            result_dict.update({
                                'confidence': confidence,
                                'variance': variance
                            })
                        else:
                            result_dict.update({
                                'confidence': 0.5,
                                'variance': 1.0
                            })
                        
                        symbol_predictions[horizon_key] = result_dict
                    
                    predictions[symbol] = symbol_predictions
                
        except Exception as e:
            logger.error(f"Error in batch price prediction: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {}

        logger.info(f'predictions ===> {predictions}')
        return predictions
    
    def _prepare_sequence(self, symbol_data: pd.DataFrame) -> Optional[torch.Tensor]:
        """
        Prepare input sequence for LSTM model.
        
        Args:
            symbol_data: DataFrame with features for single symbol
            
        Returns:
            Tensor of shape (1, sequence_length, input_size) or None if insufficient data
        """
        try:
            sequence_length = self.model_config.get('sequence_length', 60)
            
            # Get latest sequence
            latest_data = symbol_data.iloc[-sequence_length:]
            
            if len(latest_data) < sequence_length:
                logger.warning(f"Insufficient data: {len(latest_data)} < {sequence_length}")
                return None
            
            # Extract feature values
            feature_values = latest_data[self.feature_columns].values
            
            # Check for NaN values
            if np.any(np.isnan(feature_values)):
                logger.warning("NaN values found in features")
                return None
            
            # Convert to tensor and add batch dimension
            sequence_tensor = torch.FloatTensor(feature_values).unsqueeze(0)  # (1, seq_len, features)
            
            return sequence_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preparing sequence: {e}")
            return None
    
    
    def _get_fallback_prediction(self) -> Dict[str, Dict[str, float]]:
        """Get fallback prediction when normal processing fails."""
        logger.warning("Using fallback prediction due to model failure")
        fallback = {}
        for horizon in self.model.prediction_horizons:
            horizon_key = f'{horizon}min'
            # Use reasonable market-like values instead of zeros
            fallback[horizon_key] = {
                'price': 100.0,  # Reasonable stock price
                'confidence': 0.1,  # Low confidence
                'variance': 25.0  # High uncertainty
            }
        return fallback
    
    def get_model_info(self) -> Dict:
        """Get model configuration and metadata."""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'config': self.model_config,
            'prediction_horizons': self.model.prediction_horizons if self.model else [],
            'input_features': self.feature_columns,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }


