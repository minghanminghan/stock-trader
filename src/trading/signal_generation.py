#!/usr/bin/env python3
"""
Integrated Signal Generation Module

Orchestrates the complete trading pipeline from market data to executed orders.
Properly integrates LSTM predictor with trading strategy framework.

Architecture:
Market Data → LSTM Predictor → Prediction Adapter → Trading Strategy → Signal Generator → Broker

This fixes the previous architectural issue where strategy was bypassed.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.models.lstm.predictor import LSTMPredictor
# from src.trading.strategy import StrategyConfig, create_strategy, BaseStrategy
from src.alpaca.broker import AlpacaBroker, OrderRequest, create_buy_order, create_sell_order
from src.config import TICKERS, LSTM_CONFIG, STRATEGY_CONFIG
from src.utils.logging_config import logger


def 


















class PredictionAdapter:
    """
    Converts 60-minute sequence LSTM predictions to strategy-compatible format.

    Handles aggregation across the 60-minute prediction sequence and converts
    price predictions to directional signals with confidence scores.
    """

    def __init__(self,
                 horizon_weights: Optional[Dict[int, float]] = None,
                 min_return_threshold: float = 0.001):
        """
        Initialize prediction adapter.

        Args:
            horizon_weights: Weights for different minute horizons (e.g., {5: 0.3, 15: 0.4, 30: 0.2, 60: 0.1})
            min_return_threshold: Minimum return threshold for signal generation
        """
        # Default weights favoring shorter horizons
        self.horizon_weights = horizon_weights or {
            5: 0.4,   # 5-minute horizon (highest weight)
            15: 0.3,  # 15-minute horizon
            30: 0.2,  # 30-minute horizon
            60: 0.1   # 60-minute horizon (lowest weight)
        }
        self.min_return_threshold = min_return_threshold

    def convert_predictions(self,
                          lstm_predictions: Dict[str, Dict[str, np.ndarray]],
                          current_prices: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Convert LSTM 60-minute sequence predictions to strategy format.

        Args:
            lstm_predictions: LSTM output {symbol -> {'price': array[60], 'confidence': array[60]}}
            current_prices: Current prices {symbol -> price}

        Returns:
            Strategy format {symbol -> {direction, confidence, expected_return}}
        """
        adapted_predictions = {}

        for symbol, predictions in lstm_predictions.items():
            if not predictions or 'price' not in predictions:
                continue

            current_price = current_prices.get(symbol, 0.0)
            if current_price <= 0:
                continue

            price_sequence = predictions['price']
            confidence_sequence = predictions.get('confidence', np.ones_like(price_sequence) * 0.5)

            # Ensure we have valid sequences
            if len(price_sequence) != 60 or len(confidence_sequence) != 60:
                continue

            # Calculate weighted predictions across different horizons
            weighted_return = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0

            for horizon_minutes, weight in self.horizon_weights.items():
                if horizon_minutes > len(price_sequence):
                    continue

                # Get prediction at specific horizon (1-indexed to 60)
                horizon_idx = horizon_minutes - 1
                predicted_price = float(price_sequence[horizon_idx])
                confidence = float(confidence_sequence[horizon_idx])

                # Skip invalid predictions
                if np.isnan(predicted_price) or np.isnan(confidence):
                    continue

                # Calculate expected return for this horizon
                expected_return = (predicted_price - current_price) / current_price

                weighted_return += weight * expected_return
                weighted_confidence += weight * confidence
                total_weight += weight

            if total_weight == 0:
                continue

            # Normalize by total weight
            avg_return = weighted_return / total_weight
            avg_confidence = weighted_confidence / total_weight

            # Additional analysis: trend direction over the sequence
            price_trend = self._analyze_price_trend(price_sequence, current_price)
            confidence_trend = self._analyze_confidence_trend(confidence_sequence)

            # Combine weighted return with trend analysis
            final_confidence = (avg_confidence + confidence_trend) / 2
            final_return = (avg_return + price_trend) / 2

            # Determine direction based on expected return
            if abs(final_return) < self.min_return_threshold:
                direction = 'hold'
            elif final_return > 0:
                direction = 'up'
            else:
                direction = 'down'

            adapted_predictions[symbol] = {
                'direction': direction,
                'confidence': final_confidence,
                'expected_return': final_return,
                'current_price': current_price,
                'horizon_analysis': {
                    'weighted_return': avg_return,
                    'trend_return': price_trend,
                    'avg_confidence': avg_confidence,
                    'trend_confidence': confidence_trend
                }
            }

        return adapted_predictions

    def _analyze_price_trend(self, price_sequence: np.ndarray, current_price: float) -> float:
        """
        Analyze the overall trend direction of the price sequence.

        Args:
            price_sequence: Array of 60 predicted prices
            current_price: Current price for comparison

        Returns:
            Trend return (-1 to 1, negative=downtrend, positive=uptrend)
        """
        try:
            # Calculate returns across the sequence
            returns = np.diff(price_sequence) / price_sequence[:-1]

            # Calculate trend strength (average return)
            trend_return = np.mean(returns)

            # Also consider final vs current price
            final_return = (price_sequence[-1] - current_price) / current_price

            # Combine trend and final return (70% trend, 30% final)
            combined_return = 0.7 * trend_return + 0.3 * final_return

            return float(np.clip(combined_return, -1.0, 1.0))

        except Exception:
            return 0.0

    def _analyze_confidence_trend(self, confidence_sequence: np.ndarray) -> float:
        """
        Analyze the overall confidence trend.

        Args:
            confidence_sequence: Array of 60 confidence values

        Returns:
            Average confidence (0 to 1)
        """
        try:
            # Calculate average confidence, giving more weight to later predictions
            weights = np.linspace(0.5, 1.0, len(confidence_sequence))
            weighted_confidence = np.average(confidence_sequence, weights=weights)

            return float(np.clip(weighted_confidence, 0.0, 1.0))

        except Exception:
            return 0.5


class SignalGenerator:
    """
    Integrated signal generation system that combines LSTM predictions with trading strategy.
    
    Proper architecture:
    Market Data → LSTM Predictor → Prediction Adapter → Trading Strategy → Signal Generator → Broker
    
    This component orchestrates the entire pipeline from raw market data to executed orders.
    """
    
    def __init__(self, 
                 broker: AlpacaBroker,
                 strategy_config: StrategyConfig,
                 strategy_type: str,
                 model_path: str):
        """
        Initialize integrated signal generator.
        
        Args:
            broker: AlpacaBroker instance for order execution
            strategy_config: StrategyConfig with trading parameters
            strategy_type: Type of strategy to use (e.g. "momentum")
            model_path: Path to LSTM model
        """
        self.broker = broker
        self.strategy_config = strategy_config or StrategyConfig()
        
        # Initialize strategy
        self.strategy = create_strategy(strategy_type, self.strategy_config)
        
        # Initialize LSTM predictor
        try:
            self.predictor = LSTMPredictor(model_path=model_path)
            logger.info("LSTM predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LSTM predictor: {e}")
            self.predictor = None
        
        # Initialize prediction adapter for 60-minute sequence predictions
        self.adapter = PredictionAdapter(
            horizon_weights={5: 0.4, 15: 0.3, 30: 0.2, 60: 0.1},  # Weights for different minute horizons
            min_return_threshold=0.001  # strategy_config.take_profit_pct
        )
        
        # Execution tracking
        self.recent_signals = {}  # Track recent signals for rate limiting
        self.signal_history = []  # Keep history for analysis
        self.execution_stats = {
            'total_signals': 0,
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0
        }
        
        logger.info(f"Integrated SignalGenerator initialized with {strategy_type} strategy")
    
    def generate_and_execute_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete signal generation and execution pipeline.
        
        Pipeline: Market Data → LSTM → Adapter → Strategy → Orders → Broker
        
        Args:
            market_data: DataFrame with OHLCV data and MultiIndex (symbol, timestamp)
            
        Returns:
            Dictionary with execution results and signal statistics
        """
        execution_results = {
            'timestamp': datetime.now(),
            'symbols_processed': 0,
            'signals_generated': 0,
            'orders_placed': 0,
            'orders_successful': 0,
            'errors': [],
            'raw_predictions': {},
            'adapted_predictions': {},
            'strategy_signals': {},
            'orders': []
        }
        
        try:
            # Step 1: Get current prices from market data
            current_prices = self._extract_current_prices(market_data)
            execution_results['symbols_processed'] = len(current_prices)
            
            # Step 2: Generate LSTM predictions
            if self.predictor is None:
                execution_results['errors'].append("LSTM predictor not available")
                return execution_results
                
            logger.info(f"Generating LSTM predictions for {len(current_prices)} symbols")
            lstm_predictions = self.predictor.predict_prices(market_data=market_data)
            logger.info(f"LSTM predictions generated: {len(lstm_predictions)} symbols")
            
            execution_results['raw_predictions'] = lstm_predictions
            
            # Step 3: Adapt predictions to strategy format
            adapted_predictions = self.adapter.convert_predictions(
                lstm_predictions, current_prices
            )
            execution_results['adapted_predictions'] = adapted_predictions
            
            # Step 4: Generate signals using strategy
            signals = self.strategy.generate_signals(
                model_predictions=adapted_predictions,
                current_prices=current_prices
            )
            execution_results['strategy_signals'] = signals
            execution_results['signals_generated'] = len([s for s in signals.values() if s.get('action') != 'hold'])
            
            # Step 5: Process signals into orders
            orders = self.strategy.process_signals(signals)
            
            # Step 6: Execute orders through broker
            for order in orders:
                try:
                    # Execute order through broker
                    order_result = self._execute_order(order)
                    execution_results['orders'].append(order_result)
                    execution_results['orders_placed'] += 1
                    
                    if order_result.get('success', False):
                        execution_results['orders_successful'] += 1
                        self.execution_stats['successful_orders'] += 1
                    else:
                        self.execution_stats['failed_orders'] += 1
                    
                except Exception as e:
                    error_msg = f"Error executing order: {e}"
                    logger.error(error_msg)
                    execution_results['errors'].append(error_msg)
                    self.execution_stats['failed_orders'] += 1
            
            # Update statistics
            self.execution_stats['total_signals'] += execution_results['signals_generated']
            self.execution_stats['total_orders'] += execution_results['orders_placed']
            
            # Log summary
            logger.info(f"Signal generation complete: {execution_results['signals_generated']} signals, "
                       f"{execution_results['orders_placed']} orders placed")
            
        except Exception as e:
            error_msg = f"Error in signal generation pipeline: {e}"
            logger.error(error_msg)
            execution_results['errors'].append(error_msg)
        
        return execution_results
    
    def _extract_current_prices(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract current prices from market data.
        
        Args:
            market_data: DataFrame with MultiIndex (symbol, timestamp)
            
        Returns:
            Dict mapping symbol -> current price
        """
        current_prices = {}
        
        try:
            for symbol in market_data.index.get_level_values(0).unique():
                symbol_data = market_data.loc[symbol].sort_index()
                if not symbol_data.empty:
                    current_prices[symbol] = float(symbol_data.iloc[-1]['close'])
                    
        except Exception as e:
            logger.error(f"Error extracting current prices: {e}")
            
        return current_prices
    
    def _execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single order through broker.
        
        Args:
            order: Order dictionary from strategy
            
        Returns:
            Execution result dictionary
        """
        try:
            symbol: str = order.get('symbol')
            side: str = order.get('side')  # 'buy' or 'sell'
            quantity: float = order.get('quantity', 1)

            # Create order request
            if side == 'buy':
                order_request = create_buy_order(
                    symbol=symbol,
                    qty=quantity,
                )
            elif side == 'sell':
                order_request = create_sell_order(
                    symbol=symbol,
                    qty=quantity,
                )
            else:
                return {
                    'success': False,
                    'error': f"Unknown side: {side}",
                    'order': order
                }
            
            # Submit order to broker
            result = self.broker.place_order(order_request)
            
            # Record signal history
            self.signal_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_result': result
            })
            
            return {
                'success': result.get('success', False),
                'order_id': result.get('order_id'),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'broker_result': result,
                'original_order': order
            }
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return {
                'success': False,
                'error': str(e),
                'order': order
            }
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal generation and execution statistics."""
        return {
            'execution_stats': self.execution_stats.copy(),
            'recent_signals_count': len(self.recent_signals),
            'signal_history_count': len(self.signal_history),
            'strategy_status': self.strategy.get_status() if hasattr(self.strategy, 'get_status') else {},
            'model_info': self.predictor.get_model_info() if self.predictor else {}
        }
    
    def reset_statistics(self):
        """Reset signal and execution statistics."""
        self.execution_stats = {
            'total_signals': 0,
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0
        }
        self.signal_history.clear()
        self.recent_signals.clear()


def create_signal_generator(broker: AlpacaBroker, 
                           strategy_config: StrategyConfig,
                           strategy_type: str,
                           model_path: str) -> SignalGenerator:
    """
    Factory function to create configured signal generator.
    
    Args:
        broker: AlpacaBroker instance
        strategy_config: Trading strategy configuration
        strategy_type: Type of strategy ("momentum")
        model_path: Path to LSTM model
        
    Returns:
        Configured SignalGenerator instance
    """
    return SignalGenerator(
        broker=broker,
        strategy_config=strategy_config,
        strategy_type=strategy_type,
        model_path=model_path
    )


if __name__ == "__main__":
    """
    Test the integrated signal generation system.
    
    This demonstrates the complete pipeline:
    1. Create synthetic market data
    2. Initialize LSTM predictor and strategy
    3. Generate signals using proper integration
    4. Execute orders through broker
    """
    import pandas as pd
    import numpy as np
    from pprint import pprint
    from src.config import RANDOM_SEED
    from src.alpaca.broker import AlpacaBroker
    
    print("=== Testing Integrated Signal Generation System ===")
    
    # Create broker and strategy configuration
    broker = AlpacaBroker(paper=True)
    
    strategy_config = StrategyConfig(
        symbols=['SPY', 'AAPL', 'NVDA'],
        max_position_size=1000.0,
        buy_threshold=0.6,
        sell_threshold=0.6,
        max_open_positions=3
    )
    
    # Create integrated signal generator
    signal_gen = create_signal_generator(
        broker=broker,
        strategy_config=strategy_config,
        strategy_type="momentum",
        model_path="src/models/lstm/weights/best_lstm_model.pth"
    )
    
    # Create realistic market data for testing
    timestamps = pd.date_range('2024-01-01 09:30', periods=100, freq='1min')
    symbols = ['SPY', 'AAPL', 'NVDA']
    n_steps = len(timestamps)
    n_symbols = len(symbols)
    
    np.random.seed(RANDOM_SEED)
    
    # Generate vectorized random walk data with higher volatility
    price_changes = np.random.normal(0, 2.5, (n_symbols, n_steps))  # Increased from 0.5 to 2.5
    spreads = np.random.uniform(0.5, 3.0, (n_symbols, n_steps))     # Increased spread
    volumes = np.random.randint(800, 2001, (n_symbols, n_steps))
    
    # Technical indicators with higher volatility
    mom_5m = np.random.uniform(-0.05, 0.05, (n_symbols, n_steps))     # Increased from ±0.01 to ±0.05
    mom_15m = np.random.uniform(-0.03, 0.03, (n_symbols, n_steps))    # Increased from ±0.005 to ±0.03
    mom_60m = np.random.uniform(-0.08, 0.08, (n_symbols, n_steps))    # Increased from ±0.02 to ±0.08
    vol_15m = np.random.uniform(0.02, 0.08, (n_symbols, n_steps))     # Increased volatility measures
    vol_60m = np.random.uniform(0.015, 0.06, (n_symbols, n_steps))    # Increased volatility measures
    vol_zscore = np.random.uniform(-3.0, 3.0, (n_symbols, n_steps))   # Wider Z-score range
    
    # Time features
    minutes_from_start = np.arange(n_steps)
    time_sin = np.sin(2 * np.pi * minutes_from_start / (24 * 60))
    time_cos = np.cos(2 * np.pi * minutes_from_start / (24 * 60))
    
    # Build DataFrame efficiently
    data_arrays = []
    
    for i, symbol in enumerate(symbols):
        # Random walk prices with trend bias to generate more signals
        start_price = 100.0
        # Add trend bias: SPY up, AAPL down, NVDA volatile
        if symbol == 'SPY':
            trend_bias = np.linspace(0, 5, n_steps)  # Upward trend
        elif symbol == 'AAPL':  
            trend_bias = np.linspace(0, -3, n_steps)  # Downward trend
        else:  # NVDA
            trend_bias = np.sin(np.linspace(0, 4*np.pi, n_steps)) * 3  # Oscillating
            
        prices = start_price + np.cumsum(price_changes[i]) + trend_bias
        
        # OHLC data
        high_noise = np.random.uniform(0, spreads[i])
        low_noise = np.random.uniform(0, spreads[i])
        open_noise = np.random.uniform(-spreads[i]/2, spreads[i]/2)
        
        opens = prices + open_noise
        highs = prices + high_noise
        lows = prices - low_noise
        closes = prices
        
        returns_1m = np.divide(price_changes[i], np.maximum(prices - price_changes[i], 0.01))
        
        symbol_df = pd.DataFrame({
            'symbol': symbol,
            'timestamp': timestamps,
            'open': np.round(opens, 2),
            'high': np.round(highs, 2),
            'low': np.round(lows, 2),
            'close': np.round(closes, 2),
            'volume': volumes[i],
            'return_1m': np.round(returns_1m, 6),
            'mom_5m': np.round(mom_5m[i], 6),
            'mom_15m': np.round(mom_15m[i], 6),
            'mom_60m': np.round(mom_60m[i], 6),
            'vol_15m': np.round(vol_15m[i], 6),
            'vol_60m': np.round(vol_60m[i], 6),
            'vol_zscore': np.round(vol_zscore[i], 3),
            'time_sin': np.round(time_sin, 6),
            'time_cos': np.round(time_cos, 6)
        })
        data_arrays.append(symbol_df)
    
    # Create final DataFrame with MultiIndex
    market_data = pd.concat(data_arrays, ignore_index=True)
    market_data.set_index(['symbol', 'timestamp'], inplace=True)
    
    print(f"\nGenerated market data: {len(market_data)} rows, {len(symbols)} symbols")
    
    # Show price movements and volatility
    for symbol in symbols:
        symbol_data = market_data.loc[symbol]
        start_price = symbol_data.iloc[0]['close']
        end_price = symbol_data.iloc[-1]['close']
        price_change = ((end_price - start_price) / start_price) * 100
        volatility = symbol_data['return_1m'].std() * 100
        
        print(f"{symbol}: ${start_price:.2f} → ${end_price:.2f} ({price_change:+.1f}%, σ={volatility:.2f}%)")
    
    print(f"\nSample data for {symbols[0]}:")
    print(market_data.loc[symbols[0]].tail(3))
    
    # Test the integrated pipeline
    print("\n=== Running Integrated Signal Generation Pipeline ===")
    
    try:
        # Generate and execute signals
        results = signal_gen.generate_and_execute_signals(market_data)
        
        print("\n--- Execution Results ---")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Symbols processed: {results['symbols_processed']}")
        print(f"Signals generated: {results['signals_generated']}")
        print(f"Orders placed: {results['orders_placed']}")
        print(f"Orders successful: {results['orders_successful']}")
        print(f"Errors: {len(results['errors'])}")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        # Show raw predictions
        print("\n--- Raw LSTM Predictions ---")
        for symbol, predictions in results['raw_predictions'].items():
            print(f"{symbol}:")
            for horizon, pred in predictions.items():
                if isinstance(pred, dict) and 'price' in pred:
                    print(f"  {horizon}: ${pred['price']:.2f} (conf: {pred.get('confidence', 0):.3f})")
        
        # Show adapted predictions
        print("\n--- Adapted Predictions ---")
        for symbol, pred in results['adapted_predictions'].items():
            print(f"{symbol}: {pred['direction']} (conf: {pred['confidence']:.3f}, "
                  f"return: {pred['expected_return']:.4f})")
        
        # Show strategy signals
        print("\n--- Strategy Signals ---")
        if results['strategy_signals']:
            if isinstance(results['strategy_signals'], dict):
                # Show first 3 signals if it's a dictionary
                signal_items = list(results['strategy_signals'].items())[:3]
                pprint(dict(signal_items))
            else:
                # Show first 3 signals if it's a list
                pprint(results['strategy_signals'][:3])
        else:
            print("No signals generated")
        
        # Show orders
        print("\n--- Orders ---")
        if results['orders']:
            for order in results['orders']:
                print(f"Order: {order.get('symbol')} {order.get('side')} {order.get('quantity')} "
                      f"- Success: {order.get('success', False)}")
        else:
            print("No orders generated")
        
        # Get statistics
        print("\n--- Signal Statistics ---")
        stats = signal_gen.get_signal_statistics()
        pprint(stats['execution_stats'])
        
    except Exception as e:
        print(f"Error during signal generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")