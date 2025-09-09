from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import numpy as np

from src.trading.strategy import SignalData, Signal
from src.models.prediction import SignalGenerator
from src.feature_engineering import compute_features
from src.utils.logging_config import logger


class LiveSignalGenerator:
    """
    Real-time signal generator for live trading.
    Processes market data streams and generates ML-based trading signals.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize live signal generator.
        
        Args:
            model_path: Path to trained model file (uses latest if None)
        """
        self.predictor = SignalGenerator(model_path)
        self.signal_cache = {}  # Cache recent signals to avoid redundant computation
        self.last_signal_time = {}  # Track last signal generation time per symbol
        
        logger.info(f"LiveSignalGenerator initialized with model: {model_path or 'latest'}")
    
    def generate_signal_from_stream(self, symbol: str, data_stream) -> SignalData:
        """
        Generate trading signal for a single symbol from live data stream.
        
        Args:
            symbol: Stock symbol
            data_stream: LiveDataStream instance
            
        Returns:
            SignalData object with prediction, confidence, and timestamp
        """
        try:
            # Get recent market data from stream
            market_data = data_stream.get_recent_data(symbol, minutes=60)
            
            if not market_data or len(market_data) < 60:
                logger.warning(f"Insufficient data for {symbol}: {len(market_data) if market_data else 0} bars")
                return self._get_fallback_signal(symbol)
            
            # Convert to DataFrame for feature computation
            df = self._prepare_dataframe(symbol, market_data)
            
            # Compute features
            featured_df = compute_features(df)
            
            if featured_df.empty:
                logger.warning(f"Feature computation failed for {symbol}")
                return self._get_fallback_signal(symbol)
            
            # Get latest features
            latest_features = featured_df.iloc[-1]
            
            # Expected features from model training
            expected_features = [
                'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
                'vol_15m', 'vol_60m', 'vol_zscore',
                'time_sin', 'time_cos'
            ]
            
            # Check if all required features are available
            missing_features = [f for f in expected_features if f not in latest_features.index or pd.isna(latest_features[f])]
            
            if missing_features:
                logger.warning(f"Missing features for {symbol}: {missing_features}")
                return self._get_fallback_signal(symbol)
            
            # Prepare feature vector
            feature_vector = latest_features[expected_features].values.reshape(1, -1)
            
            # Generate prediction
            raw_prediction = self.predictor.model.predict(feature_vector)[0]
            probabilities = self.predictor.model.predict_proba(feature_vector)[0]
            confidence = np.max(probabilities)
            
            # Map prediction to standardized signal format
            if raw_prediction == -1:
                prediction = Signal.SELL
            elif raw_prediction == 0:
                prediction = Signal.HOLD  
            elif raw_prediction == 1:
                prediction = Signal.BUY
            else:
                prediction = Signal.HOLD  # Default fallback
            
            # Create signal data
            signal_data = SignalData(
                symbol=symbol,
                prediction=prediction,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
            
            # Update cache and tracking
            self.signal_cache[symbol] = signal_data
            self.last_signal_time[symbol] = datetime.now()
            
            logger.debug(f"Generated signal for {symbol}: {prediction.name} (confidence: {confidence:.3f})")
            return signal_data
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return self._get_fallback_signal(symbol)
    
    def generate_signals_for_symbols(self, symbols: List[str], data_stream) -> List[SignalData]:
        """
        Generate signals for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            data_stream: LiveDataStream instance
            
        Returns:
            List of SignalData objects
        """
        signals = []
        
        for symbol in symbols:
            signal_data = self.generate_signal_from_stream(symbol, data_stream)
            signals.append(signal_data)
        
        logger.debug(f"Generated {len(signals)} signals for {len(symbols)} symbols")
        return signals
    
    def get_cached_signal(self, symbol: str, max_age_minutes: int = 5) -> Optional[SignalData]:
        """
        Get cached signal if it's still fresh.
        
        Args:
            symbol: Stock symbol
            max_age_minutes: Maximum age of cached signal in minutes
            
        Returns:
            Cached SignalData if fresh, None otherwise
        """
        if symbol not in self.signal_cache or symbol not in self.last_signal_time:
            return None
        
        age = datetime.now() - self.last_signal_time[symbol]
        
        if age.total_seconds() / 60 <= max_age_minutes:
            return self.signal_cache[symbol]
        
        return None
    
    def _prepare_dataframe(self, symbol: str, market_data: List[Dict]) -> pd.DataFrame:
        """
        Convert market data to DataFrame format expected by feature engineering.
        
        Args:
            symbol: Stock symbol
            market_data: List of market data dictionaries
            
        Returns:
            Formatted DataFrame with MultiIndex (symbol, timestamp)
        """
        df = pd.DataFrame(market_data)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Set MultiIndex as expected by feature engineering
        df.set_index(['symbol', 'timestamp'], inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def _get_fallback_signal(self, symbol: str) -> SignalData:
        """
        Get fallback HOLD signal when normal processing fails.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            SignalData with HOLD prediction
        """
        return SignalData(
            symbol=symbol,
            prediction=Signal.HOLD,
            confidence=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def clear_cache(self):
        """Clear signal cache and timing data."""
        self.signal_cache.clear()
        self.last_signal_time.clear()
        logger.debug("Signal cache cleared")


class ParallelSignalGenerator(LiveSignalGenerator):
    """
    Parallel version of LiveSignalGenerator that processes multiple symbols concurrently.
    Significant performance improvement for multiple symbols.
    """
    
    def __init__(self, model_path: str = None, max_workers: int = 4):
        """
        Initialize parallel signal generator.
        
        Args:
            model_path: Path to model file (uses latest if None)
            max_workers: Maximum number of parallel workers
        """
        super().__init__(model_path)
        self.max_workers = max_workers
        self._lock = threading.Lock()  # Thread-safe cache access
        
        logger.info(f"ParallelSignalGenerator initialized with {max_workers} workers")
    
    def generate_signals_for_symbols(self, symbols: List[str], 
                                   data_stream) -> List[SignalData]:
        """
        Generate signals for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            data_stream: LiveDataStream instance
            
        Returns:
            List of SignalData objects
        """
        if len(symbols) <= 1:
            # Use parent method for single symbol
            return super().generate_signals_for_symbols(symbols, data_stream)
        
        signals = []
        
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._generate_signal_safe, symbol, data_stream): symbol 
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signal_data = future.result()
                    signals.append(signal_data)
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol}: {e}")
                    # Add fallback HOLD signal
                    signals.append(SignalData(
                        symbol=symbol,
                        prediction=Signal.HOLD,
                        confidence=0.0,
                        timestamp=datetime.now().isoformat()
                    ))
        
        # Sort signals by symbol to maintain consistent ordering
        signals.sort(key=lambda s: s.symbol)
        
        logger.debug(f"Generated {len(signals)} signals in parallel")
        return signals
    
    def _generate_signal_safe(self, symbol: str, data_stream) -> SignalData:
        """
        Thread-safe wrapper for signal generation.
        
        Args:
            symbol: Stock symbol
            data_stream: LiveDataStream instance
            
        Returns:
            SignalData object
        """
        # Check if data is fresh enough
        if not data_stream.is_data_fresh(symbol, max_age_minutes=5):
            logger.warning(f"Stale data for {symbol}, using cached signal or HOLD")
            
            # Thread-safe cache access
            with self._lock:
                if symbol in self.signal_cache:
                    return self.signal_cache[symbol]
            
            return SignalData(
                symbol=symbol,
                prediction=Signal.HOLD,
                confidence=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        # Generate fresh signal
        signal_data = self.generate_signal_from_stream(symbol, data_stream)
        
        # Thread-safe cache update
        with self._lock:
            self.signal_cache[symbol] = signal_data
        
        return signal_data


class BatchSignalProcessor:
    """
    Alternative approach: Batch process multiple symbols with single model call.
    Most efficient for large numbers of symbols.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize batch signal processor.
        
        Args:
            model_path: Path to model file
        """
        from src.models.prediction import SignalGenerator
        self.predictor = SignalGenerator(model_path)
        
        logger.info("BatchSignalProcessor initialized")
    
    def generate_batch_signals(self, symbols_data: Dict[str, List[Dict]]) -> Dict[str, SignalData]:
        """
        Process multiple symbols in a single batch operation.
        
        Args:
            symbols_data: Dict mapping symbol -> market data list
            
        Returns:
            Dict mapping symbol -> SignalData
        """
        results = {}
        
        # Prepare batch data
        valid_symbols = []
        batch_features = []
        
        for symbol, market_data in symbols_data.items():
            if len(market_data) >= 60:  # Need sufficient data for features
                try:
                    # Process individual symbol to get features
                    # (This part could be further optimized)
                    import pandas as pd
                    from src.feature_engineering import compute_features
                    
                    df = pd.DataFrame(market_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['symbol'] = symbol
                    df.set_index(['symbol', 'timestamp'], inplace=True)
                    
                    featured_df = compute_features(df)
                    
                    if not featured_df.empty:
                        latest_features = featured_df.iloc[-1]
                        expected_features = [
                            'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
                            'vol_15m', 'vol_60m', 'vol_zscore',
                            'time_sin', 'time_cos'
                        ]
                        
                        if all(feature in latest_features.index and not pd.isna(latest_features[feature]) 
                               for feature in expected_features):
                            valid_symbols.append(symbol)
                            batch_features.append(latest_features[expected_features].values)
                
                except Exception as e:
                    logger.error(f"Error processing features for {symbol}: {e}")
        
        # Batch prediction if we have valid data
        if valid_symbols and batch_features:
            try:
                import numpy as np
                X_batch = np.array(batch_features)
                
                # Single batch prediction call
                batch_predictions = self.predictor.model.predict(X_batch)
                batch_probabilities = self.predictor.model.predict_proba(X_batch)
                batch_confidences = np.max(batch_probabilities, axis=1)
                
                # Process results
                for i, symbol in enumerate(valid_symbols):
                    raw_prediction = batch_predictions[i]
                    confidence = batch_confidences[i]
                    
                    # Map prediction to standard format
                    if raw_prediction == -1:
                        prediction = Signal.SELL
                    elif raw_prediction == 0:
                        prediction = Signal.HOLD
                    elif raw_prediction == 1:
                        prediction = Signal.BUY
                    else:
                        prediction = Signal.HOLD
                    
                    results[symbol] = SignalData(
                        symbol=symbol,
                        prediction=prediction,
                        confidence=confidence,
                        timestamp=datetime.now().isoformat()
                    )
                
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
        
        # Add HOLD signals for symbols without valid predictions
        for symbol in symbols_data.keys():
            if symbol not in results:
                results[symbol] = SignalData(
                    symbol=symbol,
                    prediction=Signal.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now().isoformat()
                )
        
        logger.debug(f"Batch processed {len(results)} signals")
        return results