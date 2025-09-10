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

from src.models.lstm.model import create_model
from _utils.feature_engineering import compute_features
from src.utils.logging_config import logger


class LSTMPredictor:
    """
    LSTM model predictor for real-time trading signals.
    
    Handles model loading, data preprocessing, and multi-horizon price predictions
    with confidence estimation for trading decision making.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize LSTM predictor.
        
        Args:
            model_path: Path to saved model (.pth file). If None, loads latest model
            device: Device to run model on ('cpu', 'cuda'). If None, auto-detects
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
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
            
            # Create model with saved config
            self.model = create_model(self.model_config)
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
    
    def predict_prices(self, market_data: pd.DataFrame, 
                      return_confidence: bool = True,
                      mc_samples: int = 50) -> Dict[str, Dict[str, float]]:
        """
        Generate multi-horizon price predictions with confidence.
        
        Args:
            market_data: DataFrame with OHLCV data and MultiIndex (symbol, timestamp)
            return_confidence: Whether to compute confidence estimates
            mc_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Dict mapping symbol -> horizon -> {price, confidence, std, percentiles}
        """
        predictions = {}
        
        try:
            # Ensure we have features
            if not all(col in market_data.columns for col in self.feature_columns):
                featured_data = compute_features(market_data)
            else:
                featured_data = market_data
            
            # Process each symbol
            for symbol in featured_data.index.get_level_values(0).unique():
                symbol_data = featured_data.loc[symbol].sort_index()
                
                if len(symbol_data) < self.model_config.get('sequence_length', 60):
                    logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} bars")
                    predictions[symbol] = self._get_fallback_prediction()
                    continue
                
                # Prepare input sequence
                sequence = self._prepare_sequence(symbol_data)
                
                if sequence is None:
                    predictions[symbol] = self._get_fallback_prediction()
                    continue
                
                # Generate prediction
                symbol_predictions = self._predict_single_symbol(
                    sequence, symbol, return_confidence, mc_samples
                )
                predictions[symbol] = symbol_predictions
                
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return {}
        
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
    
    def _predict_single_symbol(self, sequence: torch.Tensor, symbol: str,
                              return_confidence: bool, mc_samples: int) -> Dict[str, Dict[str, float]]:
        """
        Generate predictions for single symbol.
        
        Args:
            sequence: Input tensor (1, seq_len, features)
            symbol: Symbol name for logging
            return_confidence: Whether to compute confidence
            mc_samples: Monte Carlo samples
            
        Returns:
            Dict mapping horizon -> prediction statistics
        """
        try:
            with torch.no_grad():
                if return_confidence and mc_samples > 1:
                    # Monte Carlo uncertainty estimation
                    results = self.model.predict_with_uncertainty(sequence, mc_samples)
                else:
                    # Standard forward pass
                    outputs = self.model(sequence, return_confidence=return_confidence)
                    results = {}
                    
                    for horizon in self.model.prediction_horizons:
                        horizon_key = f'{horizon}min'
                        price_pred = outputs[f'price_{horizon_key}'].cpu().item()
                        
                        result_dict = {'mean': price_pred, 'price': price_pred}
                        
                        if return_confidence and f'confidence_{horizon_key}' in outputs:
                            confidence = outputs[f'confidence_{horizon_key}'].cpu().item()
                            variance = outputs[f'variance_{horizon_key}'].cpu().item()
                            result_dict.update({
                                'confidence': confidence,
                                'std': np.sqrt(variance),
                                'variance': variance
                            })
                        else:
                            result_dict.update({
                                'confidence': 0.5,
                                'std': 0.0,
                                'variance': 0.0
                            })
                        
                        results[horizon_key] = result_dict
            
            logger.debug(f"Generated predictions for {symbol}: {list(results.keys())}")
            return results
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return self._get_fallback_prediction()
    
    def _get_fallback_prediction(self) -> Dict[str, Dict[str, float]]:
        """Get fallback prediction when normal processing fails."""
        fallback = {}
        for horizon in self.model.prediction_horizons:
            horizon_key = f'{horizon}min'
            fallback[horizon_key] = {
                'price': 0.0,
                'mean': 0.0,
                'confidence': 0.0,
                'std': 0.0,
                'variance': 0.0
            }
        return fallback
    
    def get_trading_signals(self, price_predictions: Dict[str, Dict[str, Dict[str, float]]],
                           current_prices: Dict[str, float],
                           min_confidence: float = 0.6,
                           min_price_change: float = 0.005) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Convert price predictions to trading signals.
        
        Args:
            price_predictions: Output from predict_prices()
            current_prices: Current market prices for each symbol
            min_confidence: Minimum confidence threshold
            min_price_change: Minimum expected price change (as fraction)
            
        Returns:
            Dict mapping symbol -> {signal, confidence, expected_return, horizon}
        """
        signals = {}
        
        for symbol, horizons in price_predictions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            best_signal = self._find_best_signal(horizons, current_price, min_confidence, min_price_change)
            signals[symbol] = best_signal
        
        return signals
    
    def _find_best_signal(self, horizons: Dict[str, Dict[str, float]], 
                         current_price: float,
                         min_confidence: float,
                         min_price_change: float) -> Dict[str, Union[str, float]]:
        """Find best trading signal across all horizons."""
        best_opportunity = {
            'signal': 'HOLD',
            'confidence': 0.0,
            'expected_return': 0.0,
            'horizon': '5min',
            'predicted_price': current_price
        }
        
        for horizon_key, prediction in horizons.items():
            if prediction['confidence'] < min_confidence:
                continue
            
            predicted_price = prediction['price']
            expected_return = (predicted_price - current_price) / current_price
            
            # Check if movement is significant enough
            if abs(expected_return) < min_price_change:
                continue
            
            # Score opportunity by confidence * expected return magnitude
            opportunity_score = prediction['confidence'] * abs(expected_return)
            current_best_score = best_opportunity['confidence'] * abs(best_opportunity['expected_return'])
            
            if opportunity_score > current_best_score:
                signal = 'BUY' if expected_return > 0 else 'SELL'
                best_opportunity = {
                    'signal': signal,
                    'confidence': prediction['confidence'],
                    'expected_return': expected_return,
                    'horizon': horizon_key,
                    'predicted_price': predicted_price
                }
        
        return best_opportunity
    
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


class ParallelLSTMPredictor(LSTMPredictor):
    """
    Parallel version of LSTM predictor for processing multiple symbols concurrently.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 max_workers: int = 4):
        """
        Initialize parallel LSTM predictor.
        
        Args:
            model_path: Path to model file
            device: Device to use
            max_workers: Maximum parallel workers
        """
        super().__init__(model_path, device)
        self.max_workers = max_workers
        logger.info(f"Parallel LSTM predictor initialized with {max_workers} workers")
    
    def predict_prices_parallel(self, market_data: Dict[str, pd.DataFrame],
                               return_confidence: bool = True,
                               mc_samples: int = 50) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Generate predictions for multiple symbols in parallel.
        
        Args:
            market_data: Dict mapping symbol -> DataFrame
            return_confidence: Whether to compute confidence
            mc_samples: Monte Carlo samples
            
        Returns:
            Combined predictions for all symbols
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        predictions = {}
        lock = threading.Lock()
        
        def predict_symbol(symbol: str, data: pd.DataFrame):
            try:
                # Create combined DataFrame with symbol index
                data_with_symbol = data.copy()
                data_with_symbol.index = pd.MultiIndex.from_product(
                    [[symbol], data.index], names=['symbol', 'timestamp']
                )
                
                symbol_predictions = self.predict_prices(
                    data_with_symbol, return_confidence, mc_samples
                )
                
                with lock:
                    predictions.update(symbol_predictions)
                
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
                with lock:
                    predictions[symbol] = self._get_fallback_prediction()
        
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(predict_symbol, symbol, data): symbol
                for symbol, data in market_data.items()
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Parallel prediction failed for {symbol}: {e}")
        
        return predictions