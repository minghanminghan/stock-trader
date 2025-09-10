#!/usr/bin/env python3
"""
Configurable Trading Strategy Framework

Provides base strategy class and momentum trading strategy implementation.
Designed to work with signal_generation module and broker integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

from src.config import TICKERS
from src.utils.logging_config import logger


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    
    # Symbol configuration
    symbols: List[str] = field(default_factory=lambda: TICKERS)
    
    # Risk management
    max_position_size: float = 1000.0  # Max dollar amount per position
    max_portfolio_risk: float = 0.02   # Max 2% portfolio risk per trade
    stop_loss_pct: float = 0.05        # 5% stop loss
    take_profit_pct: float = 0.10      # 10% take profit
    
    # Position sizing
    position_sizing_method: str = "fixed"  # "fixed", "risk_parity", "volatility_adjusted"
    base_position_size: float = 100.0      # Base position size in dollars
    
    # Strategy timing
    min_hold_time_minutes: int = 5         # Minimum hold time
    max_hold_time_minutes: int = 60        # Maximum hold time
    
    # Signal thresholds
    buy_threshold: float = 0.6             # Confidence threshold for buy signals
    sell_threshold: float = 0.6            # Confidence threshold for sell signals
    
    # Risk controls
    max_daily_trades: int = 50             # Maximum trades per day
    max_open_positions: int = 5            # Maximum simultaneous positions
    daily_loss_limit: float = 500.0       # Daily loss limit in dollars


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    side: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class StrategyState:
    """Tracks strategy execution state."""
    open_positions: Dict[str, Position] = field(default_factory=dict)
    daily_trades: int = 0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    last_reset_date: Optional[datetime] = None


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Provides common functionality for position management, risk controls,
    and integration with signal generation and broker execution.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.
        
        Args:
            config: StrategyConfig object with strategy parameters
        """
        self.config = config
        self.state = StrategyState()
        self.is_active = False
        
        logger.info(f"Strategy initialized with {len(config.symbols)} symbols")
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data: Dictionary containing market data for symbols
            
        Returns:
            Dict[symbol, signal_dict] where signal_dict contains:
            - action: 'buy', 'sell', 'hold'
            - confidence: float 0-1
            - size: position size
            - metadata: additional signal information
        """
        pass
    
    def start(self):
        """Start the strategy."""
        self.is_active = True
        self._reset_daily_state()
        logger.info("Strategy started")
    
    def stop(self):
        """Stop the strategy."""
        self.is_active = False
        logger.info("Strategy stopped")
    
    def process_signals(self, signals: Dict[str, Dict]) -> List[Dict]:
        """
        Process generated signals and create actionable orders.
        
        Args:
            signals: Dictionary of signals from generate_signals()
            
        Returns:
            List of order dictionaries ready for broker execution
        """
        if not self.is_active:
            return []
        
        self._reset_daily_state_if_needed()
        
        orders = []
        
        for symbol, signal in signals.items():
            # Skip if we've hit trading limits
            if not self._can_trade():
                break
            
            order = self._process_signal(symbol, signal)
            if order:
                orders.append(order)
        
        return orders
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update position information with current market prices.
        
        Args:
            current_prices: Dict of symbol -> current_price
        """
        for symbol, position in self.state.open_positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Calculate unrealized PnL
                if position.side == 'long':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:  # short
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
    
    def close_position(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Close a position and return order for execution.
        
        Args:
            symbol: Symbol to close
            current_price: Current market price
            
        Returns:
            Order dictionary or None
        """
        if symbol not in self.state.open_positions:
            return None
        
        position = self.state.open_positions[symbol]
        
        # Calculate realized PnL
        if position.side == 'long':
            realized_pnl = (current_price - position.entry_price) * position.quantity
            side = 'sell'
        else:  # short
            realized_pnl = (position.entry_price - current_price) * position.quantity
            side = 'buy'
        
        # Update strategy state
        self.state.daily_pnl += realized_pnl
        self.state.total_pnl += realized_pnl
        del self.state.open_positions[symbol]
        
        logger.info(f"Closing {position.side} position in {symbol}: PnL = ${realized_pnl:.2f}")
        
        return {
            'symbol': symbol,
            'side': side,
            'quantity': abs(position.quantity),
            'order_type': 'market',
            'metadata': {
                'action': 'close_position',
                'realized_pnl': realized_pnl
            }
        }
    
    def _process_signal(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """Process individual signal and create order if appropriate."""
        action = signal.get('action')
        confidence = signal.get('confidence', 0.0)
        
        # Check signal thresholds
        if action == 'buy' and confidence < self.config.buy_threshold:
            return None
        elif action == 'sell' and confidence < self.config.sell_threshold:
            return None
        elif action == 'hold':
            return None
        
        # Check if we already have a position
        if symbol in self.state.open_positions:
            return self._handle_existing_position(symbol, signal)
        
        # Check position limits
        if len(self.state.open_positions) >= self.config.max_open_positions:
            return None
        
        # Create new position order
        if action in ['buy', 'sell']:
            return self._create_position_order(symbol, signal)
        
        return None
    
    def _handle_existing_position(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """Handle signals for symbols we already have positions in."""
        position = self.state.open_positions[symbol]
        action = signal.get('action')
        
        # Check if we should close the position
        if (position.side == 'long' and action == 'sell') or \
           (position.side == 'short' and action == 'buy'):
            return {
                'symbol': symbol,
                'side': 'sell' if position.side == 'long' else 'buy',
                'quantity': abs(position.quantity),
                'order_type': 'market',
                'metadata': {
                    'action': 'close_position',
                    'signal_confidence': signal.get('confidence', 0.0)
                }
            }
        
        return None
    
    def _create_position_order(self, symbol: str, signal: Dict) -> Dict:
        """Create order for new position."""
        action = signal.get('action')
        confidence = signal.get('confidence', 0.0)
        
        # Calculate position size
        position_size = self._calculate_position_size(symbol, signal)
        
        # Create position tracking
        current_time = datetime.now()
        # Note: entry_price will be updated when order is filled
        position = Position(
            symbol=symbol,
            quantity=position_size if action == 'buy' else -position_size,
            entry_price=0.0,  # Will be updated after order execution
            entry_time=current_time,
            side='long' if action == 'buy' else 'short'
        )
        
        self.state.open_positions[symbol] = position
        self.state.daily_trades += 1
        
        return {
            'symbol': symbol,
            'side': action,
            'quantity': position_size,
            'order_type': 'market',
            'metadata': {
                'action': 'open_position',
                'signal_confidence': confidence,
                'strategy_generated': True
            }
        }
    
    def _calculate_position_size(self, symbol: str, signal: Dict) -> float:
        """Calculate appropriate position size based on risk management."""
        if self.config.position_sizing_method == "fixed":
            return self.config.base_position_size
        
        # Additional position sizing methods can be added here
        return self.config.base_position_size
    
    def _can_trade(self) -> bool:
        """Check if we can execute more trades today."""
        if self.state.daily_trades >= self.config.max_daily_trades:
            return False
        
        if self.state.daily_pnl <= -self.config.daily_loss_limit:
            return False
        
        return True
    
    def _reset_daily_state_if_needed(self):
        """Reset daily counters if it's a new trading day."""
        today = datetime.now().date()
        
        if self.state.last_reset_date != today:
            self._reset_daily_state()
    
    def _reset_daily_state(self):
        """Reset daily trading counters."""
        self.state.daily_trades = 0
        self.state.daily_pnl = 0.0
        self.state.last_reset_date = datetime.now().date()
        logger.info("Daily strategy state reset")
    
    def get_status(self) -> Dict:
        """Get current strategy status."""
        return {
            'is_active': self.is_active,
            'open_positions': len(self.state.open_positions),
            'daily_trades': self.state.daily_trades,
            'daily_pnl': self.state.daily_pnl,
            'total_pnl': self.state.total_pnl,
            'symbols_tracked': self.config.symbols,
            'position_details': {
                symbol: {
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for symbol, pos in self.state.open_positions.items()
            }
        }


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Generates signals based on price momentum and model predictions.
    Designed to work with LSTM and LightGBM model outputs.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize momentum strategy."""
        super().__init__(config)
        logger.info("MomentumStrategy initialized")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Generate momentum-based trading signals.
        
        Args:
            market_data: Dict containing:
                - model_predictions: Dict[symbol, predictions]
                - current_prices: Dict[symbol, price]
                - market_indicators: Optional market context
                
        Returns:
            Dict[symbol, signal] with trading signals
        """
        signals = {}
        
        model_predictions = market_data.get('model_predictions', {})
        current_prices = market_data.get('current_prices', {})
        
        for symbol in self.config.symbols:
            if symbol not in model_predictions:
                signals[symbol] = {'action': 'hold', 'confidence': 0.0}
                continue
            
            signal = self._generate_symbol_signal(
                symbol, 
                model_predictions[symbol],
                current_prices.get(symbol)
            )
            
            signals[symbol] = signal
        
        return signals
    
    def _generate_symbol_signal(self, symbol: str, predictions: Dict, current_price: Optional[float]) -> Dict:
        """Generate signal for individual symbol based on predictions."""
        
        # Extract prediction confidence (this will be customized based on actual model output)
        confidence = predictions.get('confidence', 0.0)
        predicted_direction = predictions.get('direction', 'hold')  # 'up', 'down', 'hold'
        
        # Convert model prediction to trading action
        if predicted_direction == 'up':
            action = 'buy'
        elif predicted_direction == 'down':
            action = 'sell'
        else:
            action = 'hold'
        
        # Apply momentum filters if needed
        signal_confidence = self._apply_momentum_filter(symbol, predictions, confidence)
        
        return {
            'action': action,
            'confidence': signal_confidence,
            'size': self.config.base_position_size,
            'metadata': {
                'strategy': 'momentum',
                'raw_prediction': predictions,
                'current_price': current_price
            }
        }
    
    def _apply_momentum_filter(self, symbol: str, predictions: Dict, base_confidence: float) -> float:
        """Apply momentum-based filters to adjust signal confidence."""
        
        # For now, return base confidence
        # Future enhancements could include:
        # - Volume confirmation
        # - Market regime detection
        # - Cross-symbol momentum correlation
        
        return base_confidence


# Factory function for easy strategy creation
def create_strategy(strategy_type: str = "momentum", config: Optional[StrategyConfig] = None) -> BaseStrategy:
    """
    Create a trading strategy instance.
    
    Args:
        strategy_type: Type of strategy to create ("momentum")
        config: Optional strategy configuration
        
    Returns:
        Strategy instance
    """
    if config is None:
        config = StrategyConfig()
    
    if strategy_type == "momentum":
        return MomentumStrategy(config)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


if __name__ == "__main__":
    # Example usage
    config = StrategyConfig(
        symbols=["SPY", "AAPL"],
        max_position_size=500.0,
        buy_threshold=0.7
    )
    
    strategy = create_strategy("momentum", config)
    
    # Example market data
    market_data = {
        'model_predictions': {
            'SPY': {'direction': 'up', 'confidence': 0.8},
            'AAPL': {'direction': 'down', 'confidence': 0.6}
        },
        'current_prices': {
            'SPY': 450.0,
            'AAPL': 175.0
        }
    }
    
    strategy.start()
    signals = strategy.generate_signals(market_data)
    orders = strategy.process_signals(signals)
    
    print("Generated signals:", signals)
    print("Generated orders:", orders)
    print("Strategy status:", strategy.get_status())
