#!/usr/bin/env python3
"""
Signal Generation Module

Connects ML model predictions to trading execution via broker integration.
Handles signal filtering and order management with extensible model interface.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

from src.models.lstm.predictor import LSTMPredictor
from src.alpaca.broker import AlpacaBroker, OrderRequest, create_buy_order, create_sell_order
from src.alpaca.data_stream import LiveDataStream
from src.config import TICKERS
from src.utils.logging_config import logger


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    
    # Model thresholds
    min_confidence: float = 0.6       # Minimum confidence to trade
    min_price_change: float = 0.005   # Minimum expected price change (0.5%)
    
    # Signal filtering
    max_signals_per_minute: int = 10  # Rate limiting
    
    # Position management
    max_position_value: float = 1000.0  # Max position size in dollars
    default_shares: int = 1            # Default number of shares to trade


class SignalGenerator:
    """
    Generates trading signals from ML model predictions and executes via broker.
    
    Integrates model predictions, applies filters, and manages
    order execution through AlpacaBroker.
    """
    
    def __init__(self, 
                 broker: AlpacaBroker,
                 config: Optional[SignalConfig] = None,
                 model_path: Optional[str] = None):
        """
        Initialize signal generator.
        
        Args:
            broker: AlpacaBroker instance for order execution
            config: SignalConfig with parameters
            model_path: Path to model (optional, defaults to LSTM)
        """
        self.broker = broker
        self.config = config or SignalConfig()
        
        # Initialize model predictor
        try:
            self.predictor = LSTMPredictor(model_path=model_path)
            logger.info("Model predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model predictor: {e}")
            self.predictor = None
        
        # Signal tracking
        self.recent_signals = {}  # Track recent signals for rate limiting
        self.signal_history = []  # Keep history for analysis
        
        logger.info("SignalGenerator initialized")
    
    def generate_and_execute_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate signals from market data and execute orders.
        
        Args:
            market_data: DataFrame with OHLCV data and MultiIndex (symbol, timestamp)
            
        Returns:
            Dictionary with execution results and signal statistics
        """
        execution_results = {
            'timestamp': datetime.now(),
            'signals_generated': 0,
            'orders_placed': 0,
            'orders_successful': 0,
            'errors': [],
            'signals': {},
            'orders': []
        }
        
        try:
            # Generate signals from models
            signals = self._generate_signals(market_data)
            execution_results['signals'] = signals
            execution_results['signals_generated'] = len([s for s in signals.values() if s['action'] != 'HOLD'])
            
            # Filter and rate limit signals
            filtered_signals = self._filter_signals(signals)
            
            # Execute orders
            orders = self._execute_signals(filtered_signals)
            execution_results['orders'] = orders
            execution_results['orders_placed'] = len(orders)
            execution_results['orders_successful'] = sum(1 for o in orders if o.get('success', False))
            
            # Log summary
            logger.info(f"Signal generation complete: {execution_results['signals_generated']} signals, "
                       f"{execution_results['orders_placed']} orders placed")
            
        except Exception as e:
            error_msg = f"Error in signal generation: {e}"
            logger.error(error_msg)
            execution_results['errors'].append(error_msg)
        
        return execution_results
    
    def _generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate trading signals from ML models.
        
        Args:
            market_data: Market data with features
            
        Returns:
            Dict mapping symbol -> signal information
        """
        signals = {}
        
        # Get current prices for signal generation
        current_prices = self._extract_current_prices(market_data)
        
        # Generate model signals
        if self.predictor:
            try:
                predictions = self.predictor.predict_prices(market_data)
                model_signals = self.predictor.get_trading_signals(
                    {symbol: predictions for symbol, predictions in predictions.items()},
                    current_prices,
                    min_confidence=self.config.min_confidence,
                    min_price_change=self.config.min_price_change
                )
                logger.debug(f"Generated model signals for {len(model_signals)} symbols")
                
                # Convert model signals to standard format
                for symbol in current_prices.keys():
                    if symbol in model_signals:
                        signal = model_signals[symbol]
                        signals[symbol] = {
                            'symbol': symbol,
                            'action': signal.get('signal', 'HOLD'),
                            'confidence': signal.get('confidence', 0.0),
                            'expected_return': signal.get('expected_return', 0.0),
                            'current_price': current_prices.get(symbol, 0.0),
                            'horizon': signal.get('horizon', '5min'),
                            'predicted_price': signal.get('predicted_price', current_prices.get(symbol, 0.0))
                        }
                    else:
                        signals[symbol] = self._get_hold_signal(symbol, current_prices.get(symbol, 0.0))
                        
            except Exception as e:
                logger.error(f"Error generating model signals: {e}")
                # Create hold signals for all symbols
                for symbol in current_prices.keys():
                    signals[symbol] = self._get_hold_signal(symbol, current_prices.get(symbol, 0.0))
        else:
            # No model available - generate hold signals
            for symbol in current_prices.keys():
                signals[symbol] = self._get_hold_signal(symbol, current_prices.get(symbol, 0.0))
        
        return signals
    
    def _extract_current_prices(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract current prices from market data."""
        current_prices = {}
        
        try:
            for symbol in market_data.index.get_level_values(0).unique():
                symbol_data = market_data.loc[symbol]
                if not symbol_data.empty:
                    # Get latest close price
                    latest_price = symbol_data.iloc[-1]['close']
                    current_prices[symbol] = float(latest_price)
        except Exception as e:
            logger.error(f"Error extracting current prices: {e}")
        
        return current_prices
    
    def _get_hold_signal(self, symbol: str, current_price: float) -> Dict:
        """
        Generate a default hold signal.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            
        Returns:
            Hold signal dictionary
        """
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 0.0,
            'expected_return': 0.0,
            'current_price': current_price,
            'horizon': '5min',
            'predicted_price': current_price
        }
    
    def _filter_signals(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Filter signals based on rate limits and other criteria.
        
        Args:
            signals: Raw signals from models
            
        Returns:
            Filtered signals ready for execution
        """
        filtered = {}
        current_time = datetime.now()
        
        # Clean old signals from recent tracking
        cutoff_time = current_time - timedelta(minutes=1)
        self.recent_signals = {
            symbol: timestamps for symbol, timestamps in self.recent_signals.items()
            if any(t > cutoff_time for t in timestamps)
        }
        
        signal_count = 0
        for symbol, signal in signals.items():
            if signal['action'] == 'HOLD':
                continue
            
            # Check rate limiting
            if signal_count >= self.config.max_signals_per_minute:
                logger.warning("Rate limit reached, skipping remaining signals")
                break
            
            # Check symbol-specific rate limiting
            recent_for_symbol = self.recent_signals.get(symbol, [])
            recent_count = len([t for t in recent_for_symbol if t > cutoff_time])
            
            if recent_count >= 3:  # Max 3 signals per symbol per minute
                logger.debug(f"Rate limit for {symbol}, skipping signal")
                continue
            
            # Check minimum confidence
            if signal['confidence'] < self.config.min_confidence:
                continue
            
            # Add to filtered signals
            filtered[symbol] = signal
            signal_count += 1
            
            # Track this signal
            if symbol not in self.recent_signals:
                self.recent_signals[symbol] = []
            self.recent_signals[symbol].append(current_time)
        
        logger.info(f"Filtered {len(filtered)} signals from {len(signals)} total")
        return filtered
    
    def _execute_signals(self, signals: Dict[str, Dict]) -> List[Dict]:
        """
        Execute trading orders based on signals.
        
        Args:
            signals: Filtered signals ready for execution
            
        Returns:
            List of order execution results
        """
        order_results = []
        
        for symbol, signal in signals.items():
            try:
                # Calculate position size
                position_size = self._calculate_position_size(signal)
                
                # Create order
                if signal['action'] == 'BUY':
                    order = create_buy_order(symbol, position_size)
                elif signal['action'] == 'SELL':
                    # Check if we have position to sell
                    position = self.broker.get_position(symbol)
                    if position and position['qty'] > 0:
                        order = create_sell_order(symbol, min(position_size, position['qty']))
                    else:
                        logger.warning(f"No position to sell for {symbol}")
                        continue
                else:
                    continue
                
                # Execute order
                result = self.broker.place_order(order)
                
                # Add signal metadata to result
                result_dict = {
                    'symbol': symbol,
                    'action': signal['action'],
                    'confidence': signal['confidence'],
                    'expected_return': signal['expected_return'],
                    'success': result.success,
                    'order_id': result.order_id,
                    'error': result.error,
                    'timestamp': datetime.now()
                }
                
                order_results.append(result_dict)
                
                # Store in signal history
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'signal': signal,
                    'order_result': result_dict
                })
                
                logger.info(f"Executed {signal['action']} for {symbol}: "
                           f"confidence={signal['confidence']:.3f}, success={result.success}")
                
            except Exception as e:
                error_msg = f"Error executing signal for {symbol}: {e}"
                logger.error(error_msg)
                order_results.append({
                    'symbol': symbol,
                    'action': signal['action'],
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now()
                })
        
        return order_results
    
    def _calculate_position_size(self, signal: Dict) -> float:
        """
        Calculate position size based on signal and risk management.
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size (number of shares)
        """
        # Simple fixed size for now
        # Could be enhanced with:
        # - Risk-based sizing
        # - Confidence-based sizing
        # - Available buying power consideration
        
        return float(self.config.default_shares)
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent signal generation."""
        if not self.signal_history:
            return {
                'total_signals': 0,
                'successful_orders': 0,
                'success_rate': 0.0,
                'recent_symbols': []
            }
        
        recent_history = [
            h for h in self.signal_history 
            if h['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        successful_orders = sum(1 for h in recent_history if h['order_result']['success'])
        total_signals = len(recent_history)
        
        return {
            'total_signals': total_signals,
            'successful_orders': successful_orders,
            'success_rate': successful_orders / max(total_signals, 1),
            'recent_symbols': list(set(h['signal']['symbol'] for h in recent_history[-10:])),
            'avg_confidence': sum(h['signal']['confidence'] for h in recent_history) / max(total_signals, 1),
            'horizons_used': list(set(h['signal'].get('horizon', '5min') for h in recent_history))
        }


class LiveSignalGenerator(SignalGenerator):
    """
    Live signal generator that integrates with data streaming.
    
    Connects to LiveDataStream for real-time market data and generates
    signals automatically.
    """
    
    def __init__(self,
                 broker: AlpacaBroker,
                 data_stream: LiveDataStream,
                 config: Optional[SignalConfig] = None,
                 model_path: Optional[str] = None,
                 update_interval_seconds: int = 60):
        """
        Initialize live signal generator.
        
        Args:
            broker: AlpacaBroker for order execution
            data_stream: LiveDataStream for market data
            config: SignalConfig parameters
            model_path: Path to model
            update_interval_seconds: How often to generate signals
        """
        super().__init__(broker, config, model_path)
        
        self.data_stream = data_stream
        self.update_interval = update_interval_seconds
        self.is_running = False
        
        # Subscribe to data updates
        self.data_stream.add_subscriber(self._on_new_data)
        
        logger.info(f"LiveSignalGenerator initialized with {update_interval}s update interval")
    
    def start(self, symbols: Optional[List[str]] = None):
        """
        Start live signal generation.
        
        Args:
            symbols: List of symbols to track (default: from config)
        """
        if symbols is None:
            symbols = TICKERS
        
        self.is_running = True
        self.data_stream.start_stream(symbols)
        
        logger.info(f"Live signal generation started for {symbols}")
    
    def stop(self):
        """Stop live signal generation."""
        self.is_running = False
        self.data_stream.stop_stream()
        logger.info("Live signal generation stopped")
    
    def _on_new_data(self, symbol: str, bar_data: Dict):
        """
        Callback for new market data.
        
        Args:
            symbol: Symbol that was updated
            bar_data: New bar data
        """
        # This could trigger signal generation, but for now we'll use 
        # scheduled updates to avoid too frequent trading
        pass


# Convenience function for easy setup
def create_signal_generator(broker: AlpacaBroker, 
                          config: Optional[SignalConfig] = None,
                          live_stream: Optional[LiveDataStream] = None) -> SignalGenerator:
    """
    Create a signal generator instance.
    
    Args:
        broker: AlpacaBroker instance
        config: SignalConfig (optional)
        live_stream: LiveDataStream for real-time signals (optional)
        
    Returns:
        SignalGenerator instance
    """
    if live_stream:
        return LiveSignalGenerator(broker, live_stream, config)
    else:
        return SignalGenerator(broker, config)


if __name__ == "__main__":
    # Example usage
    from src.alpaca.broker import AlpacaBroker
    
    # Create broker and signal generator
    broker = AlpacaBroker(paper=True)
    signal_gen = create_signal_generator(broker)
    
    # Example market data (in practice, this would come from data stream)
    import pandas as pd
    
    # Create sample data
    timestamps = pd.date_range('2024-01-01 09:30', periods=100, freq='1min')
    symbols = ['SPY', 'AAPL']
    
    data_list = []
    for symbol in symbols:
        for timestamp in timestamps:
            data_list.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000,
                'return_1m': 0.001,
                'mom_5m': 0.005,
                'mom_15m': 0.002,
                'mom_60m': 0.01,
                'vol_15m': 0.02,
                'vol_60m': 0.015,
                'vol_zscore': 0.5,
                'time_sin': 0.0,
                'time_cos': 1.0
            })
    
    market_data = pd.DataFrame(data_list)
    market_data.set_index(['symbol', 'timestamp'], inplace=True)
    
    # Generate and execute signals
    results = signal_gen.generate_and_execute_signals(market_data)
    print("Signal generation results:", results)
    print("Signal statistics:", signal_gen.get_signal_statistics())