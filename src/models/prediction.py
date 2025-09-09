import pickle
import os
import glob
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

from src.feature_engineering import compute_features
from src.utils.logging_config import logger


class ModelLoader:
    """
    Utility class for loading and managing trained models.
    Handles model loading from weights directory and caching.
    """
    
    def __init__(self, weights_dir: str = "src/models/weights"):
        """
        Initialize model loader.
        
        Args:
            weights_dir: Directory containing model weight files
        """
        self.weights_dir = weights_dir
        self.loaded_models: Dict[str, object] = {}  # Cache loaded models
        
        logger.info(f"ModelLoader initialized - weights dir: {weights_dir}")
    
    def get_latest_model_path(self) -> Optional[str]:
        """
        Find the most recently saved model file.
        
        Returns:
            Path to latest model file or None if not found
        """
        pattern = os.path.join(self.weights_dir, "lgbm_model_*.pkl")
        model_files = glob.glob(pattern)
        
        if not model_files:
            logger.error(f"No model files found in {self.weights_dir}")
            return None
        
        # Sort by timestamp in filename (YYYYMMDD_HHMMSS format)
        model_files.sort(key=lambda x: os.path.basename(x).split('_')[-1])
        latest_model = model_files[-1]
        
        logger.info(f"Latest model found: {os.path.basename(latest_model)}")
        return latest_model
    
    def load_model(self, model_path: str = None) -> Optional[object]:
        """
        Load model from pickle file.
        
        Args:
            model_path: Path to model file (uses latest if None)
            
        Returns:
            Loaded model or None if failed
        """
        if model_path is None:
            model_path = self.get_latest_model_path()
            
        if model_path is None:
            return None
        
        # Check if already cached
        if model_path in self.loaded_models:
            logger.debug(f"Using cached model: {os.path.basename(model_path)}")
            return self.loaded_models[model_path]
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Cache the loaded model
            self.loaded_models[model_path] = model
            
            logger.info(f"Model loaded successfully: {os.path.basename(model_path)}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def list_available_models(self) -> List[Dict]:
        """
        List all available model files with metadata.
        
        Returns:
            List of model info dictionaries
        """
        pattern = os.path.join(self.weights_dir, "lgbm_model_*.pkl")
        model_files = glob.glob(pattern)
        
        models = []
        for model_file in model_files:
            filename = os.path.basename(model_file)
            timestamp_str = filename.replace("lgbm_model_", "").replace(".pkl", "")
            
            try:
                # Parse timestamp from filename
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                file_size = os.path.getsize(model_file)
                
                models.append({
                    'path': model_file,
                    'filename': filename,
                    'timestamp': timestamp,
                    'size_bytes': file_size,
                    'age_hours': (datetime.now() - timestamp).total_seconds() / 3600
                })
            except ValueError:
                logger.warning(f"Could not parse timestamp from filename: {filename}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models


class SignalGenerator:
    """
    ML prediction engine that generates trading signals from market data.
    Uses pre-trained models to classify market direction.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize signal generator.
        
        Args:
            model_path: Path to model file (uses latest if None)
        """
        self.model_loader = ModelLoader()
        self.model = None
        self.model_path = model_path
        
        # Feature names expected by the model (from training)
        self.expected_features = [
            'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
            'vol_15m', 'vol_60m', 'vol_zscore',
            'time_sin', 'time_cos'
        ]
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the ML model.
        
        Returns:
            True if model loaded successfully
        """
        self.model = self.model_loader.load_model(self.model_path)
        
        if self.model is None:
            logger.error("Failed to load ML model")
            return False
        
        logger.info("ML model loaded and ready for predictions")
        return True
    
    def generate_signal(self, symbol: str, market_data: List[Dict]) -> Dict:
        """
        Generate trading signal from market data.
        
        Args:
            symbol: Stock symbol
            market_data: List of OHLCV bar dictionaries (recent history)
            
        Returns:
            Signal dictionary with prediction and confidence
        """
        if self.model is None:
            logger.error("No model loaded")
            return {
                'symbol': symbol,
                'prediction': 1,  # HOLD
                'confidence': 0.0,
                'error': 'No model loaded'
            }
        
        if len(market_data) < 60:  # Need at least 60 bars for features
            logger.warning(f"Insufficient data for {symbol}: {len(market_data)} bars")
            return {
                'symbol': symbol,
                'prediction': 1,  # HOLD
                'confidence': 0.0,
                'error': 'Insufficient data'
            }
        
        try:
            # Convert market data to DataFrame
            df = pd.DataFrame(market_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set multi-index as expected by feature engineering
            df['symbol'] = symbol
            df.set_index(['symbol', 'timestamp'], inplace=True)
            
            # Compute features
            featured_df = compute_features(df)
            
            if featured_df.empty:
                return {
                    'symbol': symbol,
                    'prediction': 1,  # HOLD
                    'confidence': 0.0,
                    'error': 'Feature computation failed'
                }
            
            # Get the latest features (most recent row)
            latest_features = featured_df.iloc[-1]
            
            # Check for missing features
            missing_features = []
            for feature in self.expected_features:
                if feature not in latest_features.index or pd.isna(latest_features[feature]):
                    missing_features.append(feature)
            
            if missing_features:
                logger.warning(f"Missing features for {symbol}: {missing_features}")
                return {
                    'symbol': symbol,
                    'prediction': 1,  # HOLD
                    'confidence': 0.0,
                    'error': f'Missing features: {missing_features}'
                }
            
            # Extract feature vector
            X = latest_features[self.expected_features].values.reshape(1, -1)
            
            # Make prediction
            raw_prediction = self.model.predict(X)[0]
            prediction_proba = self.model.predict_proba(X)[0]
            
            # Map model predictions to standardized signals
            # Model outputs: {-1: DOWN, 0: FLAT, 1: UP}
            # Convert to: {0: SELL, 1: HOLD, 2: BUY}
            if raw_prediction == -1:
                prediction = 0  # DOWN -> SELL
            elif raw_prediction == 0:
                prediction = 1  # FLAT -> HOLD
            elif raw_prediction == 1:
                prediction = 2  # UP -> BUY
            else:
                prediction = 1  # Default to HOLD for unexpected values
            
            # Get confidence (max probability)
            confidence = float(np.max(prediction_proba))
            
            # Signal mapping: 0=SELL, 1=HOLD, 2=BUY
            signal_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            
            logger.debug(f"Signal for {symbol}: {signal_names[prediction]} "
                        f"(confidence: {confidence:.3f})")
            
            return {
                'symbol': symbol,
                'prediction': int(prediction),
                'confidence': confidence,
                'probabilities': {
                    'sell': float(prediction_proba[0]) if len(prediction_proba) > 0 else 0.33,
                    'hold': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.33,
                    'buy': float(prediction_proba[2]) if len(prediction_proba) > 2 else 0.34
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'prediction': 1,  # HOLD
                'confidence': 0.0,
                'error': str(e)
            }
    
    def generate_signals_batch(self, symbols_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Generate signals for multiple symbols.
        
        Args:
            symbols_data: Dict mapping symbol -> market data list
            
        Returns:
            Dict mapping symbol -> signal dict
        """
        signals = {}
        
        for symbol, market_data in symbols_data.items():
            signals[symbol] = self.generate_signal(symbol, market_data)
        
        return signals
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {'loaded': False, 'error': 'No model loaded'}
        
        model_path = self.model_path or self.model_loader.get_latest_model_path()
        
        info = {
            'loaded': True,
            'model_path': model_path,
            'model_type': type(self.model).__name__,
            'expected_features': self.expected_features,
            'n_features': len(self.expected_features)
        }
        
        # Try to get model-specific info
        try:
            if hasattr(self.model, 'n_classes_'):
                info['n_classes'] = self.model.n_classes_
            if hasattr(self.model, 'feature_importances_'):
                info['feature_importances'] = dict(zip(
                    self.expected_features, 
                    self.model.feature_importances_
                ))
        except:
            pass
        
        return info