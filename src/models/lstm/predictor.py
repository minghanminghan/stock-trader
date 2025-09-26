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

from src.config import LSTM_CONFIG
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
            self.model_config = checkpoint.get('config', None)
            if self.model_config is None:
                raise ValueError(f"No model config found in checkpoint: {self.model_path}")

            # Create model with saved/inferred config
            self.model = StockPriceLSTM(**LSTM_CONFIG['model'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded LSTM model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
            logger.info(f"Model prediction horizon: {self.model.prediction_horizon} minutes (60-element sequence)")
            
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
    
    def predict_prices(self, market_data: pd.DataFrame) -> Dict[str, Dict[str, any]]:
        """
        Generate 60-minute sequence price predictions with confidence using batch processing.

        Args:
            market_data: DataFrame with OHLCV data and MultiIndex (symbol, timestamp)

        Returns:
            Dict mapping symbol -> {price: array[60], confidence: array[60], variance: array[60]}
        """
        predictions = {}
        
        try:
            featured_data = compute_features(market_data)
            
            # Collect sequences for batch processing
            sequences = []
            symbols = []
            
            for symbol in featured_data.index.get_level_values(0).unique():
                symbol_data = featured_data.loc[symbol].sort_index()
                
                if len(symbol_data) < self.model_config.get('sequence_length', 60):
                    logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} bars")
                    # continue
                
                # Prepare input sequence
                sequence = self._prepare_sequence(symbol_data)

                sequences.append(sequence)
                symbols.append(symbol)
            
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
                    # Extract 60-element sequences for this symbol (these are percentage changes)
                    pct_change_sequence = batch_outputs['price'][i].cpu().numpy()  # Shape: (60,)
                    confidence_sequence = batch_outputs['confidence'][i].cpu().numpy()  # Shape: (60,)
                    variance_sequence = batch_outputs['variance'][i].cpu().numpy()  # Shape: (60,)

                    # Get current price for this symbol to convert percentage changes back to absolute prices
                    current_price = self._extract_current_prices({symbol: featured_data.loc[symbol]})[symbol]

                    # Convert percentage changes back to absolute prices
                    # pct_change is in percentage scale (multiply by 100 in training)
                    # price = current_price * (1 + pct_change/100)
                    price_sequence = current_price * (1 + pct_change_sequence / 100)

                    predictions[symbol] = {
                        'price': price_sequence,
                        'confidence': confidence_sequence,
                        'variance': variance_sequence
                    }
                
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
            Tensor of shape (1, sequence_length, input_size)
        """
        try:
            sequence_length = self.model_config.get('sequence_length', 60)
            
            # Get latest sequence
            latest_data = symbol_data.iloc[-sequence_length:]
            
            if len(latest_data) < sequence_length:
                logger.warning(f"Insufficient data: {len(latest_data)} < {sequence_length}")
            
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

    def _extract_current_prices(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract current prices from market data for a subset of symbols."""
        current_prices = {}
        for symbol, symbol_data in market_data.items():
            if not symbol_data.empty:
                current_prices[symbol] = float(symbol_data.iloc[-1]['close'])
        return current_prices

    def get_model_info(self) -> Dict:
        """Get model configuration and metadata."""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'config': self.model_config,
            'prediction_horizon': self.model.prediction_horizon,
            'input_features': self.feature_columns,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }


if __name__ == "__main__":
    """
    Unit tests for LSTMPredictor methods.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    import tempfile
    import os

    print("=== LSTM Predictor Unit Tests ===\n")

    # Create test market data
    def create_test_market_data():
        """Create realistic test market data."""
        timestamps = pd.date_range('2024-01-01 09:30', periods=100, freq='1min')
        symbols = ['AAPL', 'MSFT']

        data_arrays = []
        np.random.seed(42)

        for symbol in symbols:
            prices = 100 + np.cumsum(np.random.normal(0, 0.5, len(timestamps)))
            spreads = np.random.uniform(0.1, 0.5, len(timestamps))

            symbol_df = pd.DataFrame({
                'symbol': symbol,
                'timestamp': timestamps,
                'open': prices + np.random.uniform(-spreads/2, spreads/2),
                'high': prices + spreads,
                'low': prices - spreads,
                'close': prices,
                'volume': np.random.randint(1000, 5000, len(timestamps)),
                'return_1m': np.random.normal(0, 0.01, len(timestamps)),
                'mom_5m': np.random.normal(0, 0.005, len(timestamps)),
                'mom_15m': np.random.normal(0, 0.003, len(timestamps)),
                'mom_60m': np.random.normal(0, 0.01, len(timestamps)),
                'vol_15m': np.random.uniform(0.005, 0.02, len(timestamps)),
                'vol_60m': np.random.uniform(0.003, 0.015, len(timestamps)),
                'vol_zscore': np.random.normal(0, 1, len(timestamps)),
                'time_sin': np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 60)),
                'time_cos': np.cos(2 * np.pi * np.arange(len(timestamps)) / (24 * 60))
            })
            data_arrays.append(symbol_df)

        market_data = pd.concat(data_arrays, ignore_index=True)
        market_data.set_index(['symbol', 'timestamp'], inplace=True)
        return market_data


    try:
        print("1. Testing _extract_current_prices method...")
        # Create test data
        market_data = create_test_market_data()

        # Test _extract_current_prices with dict format
        test_data_dict = {}
        for symbol in ['AAPL', 'MSFT']:
            test_data_dict[symbol] = market_data.loc[symbol]

        # Load actual trained model weights
        model_path = "src/models/lstm/weights/best_log_lstm_model.pth"
        try:
            predictor = LSTMPredictor(model_path=model_path)

            # Test current price extraction
            current_prices = predictor._extract_current_prices(test_data_dict)
            print(f"✓ Current prices extracted: {current_prices}")
            assert len(current_prices) == 2, "Should extract prices for both symbols"
            assert all(price > 0 for price in current_prices.values()), "All prices should be positive"

        except Exception as e:
            print(f"✗ Error testing current price extraction: {e}")

        print("\n2. Testing _prepare_sequence method...")
        try:
            # Test with sufficient data
            symbol_data = market_data.loc['AAPL']
            sequence = predictor._prepare_sequence(symbol_data)

            if sequence is not None:
                print(f"✓ Sequence prepared successfully, shape: {sequence.shape}")
                expected_shape = (1, 60, len(predictor.feature_columns))
                assert sequence.shape == expected_shape, f"Expected shape {expected_shape}, got {sequence.shape}"
            else:
                print("✗ Sequence preparation returned None")

        except Exception as e:
            print(f"✗ Error testing sequence preparation: {e}")

        print("\n3. Testing get_model_info method...")
        try:
            model_info = predictor.get_model_info()
            print(f"✓ Model info keys: {list(model_info.keys())}")
            required_keys = ['model_path', 'device', 'config', 'prediction_horizon', 'input_features', 'model_parameters']
            for key in required_keys:
                assert key in model_info, f"Missing required key: {key}"
            print(f"✓ All required model info keys present")
            print(f"  - Device: {model_info['device']}")
            print(f"  - Prediction horizon: {model_info['prediction_horizon']}")
            print(f"  - Input features count: {len(model_info['input_features'])}")

        except Exception as e:
            print(f"✗ Error testing model info: {e}")

        print("\n4. Testing predict_prices method...")
        try:
            # Test with actual trained model weights
            predictions = predictor.predict_prices(market_data)

            if predictions:
                print(f"✓ Predictions generated for symbols: {list(predictions.keys())}")
                for symbol, pred in predictions.items():
                    print(f"  - {symbol}: {list(pred.keys())}")
                    if 'price' in pred:
                        print(f"    Price array shape: {pred['price'].shape}")
                        print(f"    Sample predictions: {pred['price'][:5]}")  # Show first 5 predictions
                        print(f"    Confidence range: {pred['confidence'].min():.3f} - {pred['confidence'].max():.3f}")
            else:
                print("! No predictions generated")

        except Exception as e:
            print(f"✗ Error in predict_prices: {type(e).__name__}: {e}")

        print("\n=== Unit Tests Complete ===")
        print("Tests run with actual trained model weights")

    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

