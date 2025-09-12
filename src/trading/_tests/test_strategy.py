#!/usr/bin/env python3
"""
Unit tests for src/trading/strategy.py - Trading Strategy Framework
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, date
from typing import Dict, Any

from src.trading.strategy import (
    StrategyConfig, Position, StrategyState, BaseStrategy, 
    MomentumStrategy, create_strategy
)


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StrategyConfig()
        
        # Check default values
        assert isinstance(config.symbols, list)
        assert config.max_position_size == 1000.0
        assert config.max_portfolio_risk == 0.02
        assert config.stop_loss_pct == 0.05
        assert config.take_profit_pct == 0.10
        assert config.position_sizing_method == "fixed"
        assert config.base_position_size == 100.0
        assert config.min_hold_time_minutes == 5
        assert config.max_hold_time_minutes == 60
        assert config.buy_threshold == 0.5
        assert config.sell_threshold == 0.5
        assert config.max_daily_trades == 50
        assert config.max_open_positions == 5
        assert config.daily_loss_limit == 500.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_symbols = ["SPY", "AAPL"]
        config = StrategyConfig(
            symbols=custom_symbols,
            max_position_size=2000.0,
            buy_threshold=0.7,
            max_daily_trades=10
        )
        
        assert config.symbols == custom_symbols
        assert config.max_position_size == 2000.0
        assert config.buy_threshold == 0.7
        assert config.max_daily_trades == 10
        # Other values should remain default
        assert config.sell_threshold == 0.5


class TestPosition:
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test position creation."""
        entry_time = datetime.now()
        position = Position(
            symbol="SPY",
            quantity=10.0,
            entry_price=100.0,
            entry_time=entry_time,
            side="long",
            stop_loss=95.0,
            take_profit=110.0
        )
        
        assert position.symbol == "SPY"
        assert position.quantity == 10.0
        assert position.entry_price == 100.0
        assert position.entry_time == entry_time
        assert position.side == "long"
        assert position.stop_loss == 95.0
        assert position.take_profit == 110.0
        assert position.unrealized_pnl == 0.0
    
    def test_position_defaults(self):
        """Test position with default values."""
        position = Position(
            symbol="AAPL",
            quantity=5.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            side="short"
        )
        
        assert position.stop_loss is None
        assert position.take_profit is None
        assert position.unrealized_pnl == 0.0


class TestStrategyState:
    """Test StrategyState dataclass."""
    
    def test_default_state(self):
        """Test default strategy state."""
        state = StrategyState()
        
        assert isinstance(state.open_positions, dict)
        assert len(state.open_positions) == 0
        assert state.daily_trades == 0
        assert state.daily_pnl == 0.0
        assert state.total_pnl == 0.0
        assert state.last_reset_date is None
    
    def test_state_with_positions(self):
        """Test strategy state with positions."""
        position = Position(
            symbol="SPY",
            quantity=10.0,
            entry_price=100.0,
            entry_time=datetime.now(),
            side="long"
        )
        
        state = StrategyState(
            open_positions={"SPY": position},
            daily_trades=5,
            daily_pnl=50.0,
            total_pnl=200.0
        )
        
        assert "SPY" in state.open_positions
        assert state.daily_trades == 5
        assert state.daily_pnl == 50.0
        assert state.total_pnl == 200.0


class TestBaseStrategy:
    """Test BaseStrategy abstract base class."""
    
    class ConcreteStrategy(BaseStrategy):
        """Concrete implementation for testing."""
        
        def generate_signals(self, model_predictions: Dict[str, Dict[str, Any]], 
                           current_prices: Dict[str, float]) -> Dict[str, Dict]:
            return {
                symbol: {'action': 'hold', 'confidence': 0.5}
                for symbol in self.config.symbols
            }
    
    @pytest.fixture
    def strategy_config(self):
        """Create test strategy configuration."""
        return StrategyConfig(
            symbols=["SPY", "AAPL"],
            max_position_size=1000.0,
            buy_threshold=0.6,
            sell_threshold=0.6,
            max_daily_trades=10,
            max_open_positions=3
        )
    
    @pytest.fixture
    def strategy(self, strategy_config):
        """Create test strategy instance."""
        return self.ConcreteStrategy(strategy_config)
    
    def test_strategy_initialization(self, strategy, strategy_config):
        """Test strategy initialization."""
        assert strategy.config == strategy_config
        assert isinstance(strategy.state, StrategyState)
        assert strategy.is_active is False
        assert len(strategy.state.open_positions) == 0
    
    def test_start_stop_strategy(self, strategy):
        """Test starting and stopping strategy."""
        # Start strategy
        strategy.start()
        assert strategy.is_active is True
        
        # Stop strategy
        strategy.stop()
        assert strategy.is_active is False
    
    def test_update_positions_long(self, strategy):
        """Test updating long positions."""
        # Add a long position
        position = Position(
            symbol="SPY",
            quantity=10.0,
            entry_price=100.0,
            entry_time=datetime.now(),
            side="long"
        )
        strategy.state.open_positions["SPY"] = position
        
        # Update with new price
        current_prices = {"SPY": 105.0}
        strategy.update_positions(current_prices)
        
        # Check unrealized PnL calculation
        expected_pnl = (105.0 - 100.0) * 10.0  # $50
        assert strategy.state.open_positions["SPY"].unrealized_pnl == expected_pnl
    
    def test_update_positions_short(self, strategy):
        """Test updating short positions."""
        # Add a short position
        position = Position(
            symbol="AAPL",
            quantity=-5.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            side="short"
        )
        strategy.state.open_positions["AAPL"] = position
        
        # Update with new price (price went down - profitable for short)
        current_prices = {"AAPL": 145.0}
        strategy.update_positions(current_prices)
        
        # Check unrealized PnL calculation for short position
        expected_pnl = (150.0 - 145.0) * 5.0  # $25 profit
        assert strategy.state.open_positions["AAPL"].unrealized_pnl == expected_pnl
    
    def test_close_position_long(self, strategy):
        """Test closing long position."""
        # Add a long position
        position = Position(
            symbol="SPY",
            quantity=10.0,
            entry_price=100.0,
            entry_time=datetime.now(),
            side="long"
        )
        strategy.state.open_positions["SPY"] = position
        
        # Close position at profit
        current_price = 105.0
        order = strategy.close_position("SPY", current_price)
        
        assert order is not None
        assert order['symbol'] == "SPY"
        assert order['side'] == "sell"
        assert order['quantity'] == 10.0
        
        # Position should be removed and PnL updated
        assert "SPY" not in strategy.state.open_positions
        assert strategy.state.daily_pnl == 50.0  # (105 - 100) * 10
        assert strategy.state.total_pnl == 50.0
    
    def test_close_position_short(self, strategy):
        """Test closing short position."""
        # Add a short position
        position = Position(
            symbol="AAPL",
            quantity=-5.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            side="short"
        )
        strategy.state.open_positions["AAPL"] = position
        
        # Close position at profit (price went down)
        current_price = 145.0
        order = strategy.close_position("AAPL", current_price)
        
        assert order is not None
        assert order['symbol'] == "AAPL"
        assert order['side'] == "buy"  # Buy to close short
        assert order['quantity'] == 5.0  # Absolute value
        
        # Check realized PnL
        assert strategy.state.daily_pnl == 25.0  # (150 - 145) * 5
    
    def test_close_nonexistent_position(self, strategy):
        """Test closing position that doesn't exist."""
        order = strategy.close_position("NONEXISTENT", 100.0)
        
        assert order is None
    
    def test_process_signal_buy(self, strategy):
        """Test processing buy signal."""
        signal = {
            'action': 'buy',
            'confidence': 0.8,
            'size': 100.0
        }
        
        order = strategy._process_signal("SPY", signal)
        
        assert order is not None
        assert order['symbol'] == "SPY"
        assert order['side'] == "buy"
        assert order['quantity'] == 100.0
        assert "SPY" in strategy.state.open_positions
    
    def test_process_signal_sell(self, strategy):
        """Test processing sell signal."""
        signal = {
            'action': 'sell',
            'confidence': 0.8,
            'size': 100.0
        }
        
        order = strategy._process_signal("AAPL", signal)
        
        assert order is not None
        assert order['symbol'] == "AAPL"
        assert order['side'] == "sell"
        assert order['quantity'] == 100.0
        assert "AAPL" in strategy.state.open_positions
    
    def test_process_signal_hold(self, strategy):
        """Test processing hold signal."""
        signal = {
            'action': 'hold',
            'confidence': 0.5
        }
        
        order = strategy._process_signal("SPY", signal)
        
        assert order is None
    
    def test_process_signal_low_confidence(self, strategy):
        """Test processing signal with low confidence."""
        signal = {
            'action': 'buy',
            'confidence': 0.3  # Below buy_threshold of 0.6
        }
        
        order = strategy._process_signal("SPY", signal)
        
        assert order is None
    
    def test_process_signal_existing_position(self, strategy):
        """Test processing signal when position already exists."""
        # Add existing long position
        position = Position(
            symbol="SPY",
            quantity=10.0,
            entry_price=100.0,
            entry_time=datetime.now(),
            side="long"
        )
        strategy.state.open_positions["SPY"] = position
        
        # Send sell signal (should close position)
        signal = {
            'action': 'sell',
            'confidence': 0.8
        }
        
        order = strategy._process_signal("SPY", signal)
        
        assert order is not None
        assert order['side'] == "sell"  # Close long position
    
    def test_process_signal_position_limit(self, strategy):
        """Test processing signal when position limit is reached."""
        # Fill up position slots
        for i in range(strategy.config.max_open_positions):
            position = Position(
                symbol=f"STOCK{i}",
                quantity=10.0,
                entry_price=100.0,
                entry_time=datetime.now(),
                side="long"
            )
            strategy.state.open_positions[f"STOCK{i}"] = position
        
        # Try to open another position
        signal = {
            'action': 'buy',
            'confidence': 0.8
        }
        
        order = strategy._process_signal("NEWSTOCK", signal)
        
        assert order is None  # Should be rejected due to position limit
    
    def test_can_trade_limits(self, strategy):
        """Test trading limits."""
        # Initially should be able to trade
        assert strategy._can_trade() is True
        
        # Hit daily trade limit
        strategy.state.daily_trades = strategy.config.max_daily_trades
        assert strategy._can_trade() is False
        
        # Reset trades but hit loss limit
        strategy.state.daily_trades = 0
        strategy.state.daily_pnl = -strategy.config.daily_loss_limit - 1
        assert strategy._can_trade() is False
    
    def test_daily_reset(self, strategy):
        """Test daily state reset."""
        # Set some daily state
        strategy.state.daily_trades = 10
        strategy.state.daily_pnl = -100.0
        strategy.state.last_reset_date = date.today()
        
        # Reset manually
        strategy._reset_daily_state()
        
        assert strategy.state.daily_trades == 0
        assert strategy.state.daily_pnl == 0.0
        assert strategy.state.last_reset_date == date.today()
    
    def test_daily_reset_if_needed(self, strategy):
        """Test automatic daily reset."""
        # Set state for previous day
        from datetime import timedelta
        yesterday = date.today() - timedelta(days=1)
        strategy.state.daily_trades = 10
        strategy.state.daily_pnl = -100.0
        strategy.state.last_reset_date = yesterday
        
        # Process signals (should trigger reset)
        signals = {"SPY": {"action": "hold", "confidence": 0.5}}
        strategy.start()
        orders = strategy.process_signals(signals)
        
        # Should have reset daily state
        assert strategy.state.daily_trades == 0
        assert strategy.state.daily_pnl == 0.0
        assert strategy.state.last_reset_date == date.today()
    
    def test_get_status(self, strategy):
        """Test status reporting."""
        # Add a position
        position = Position(
            symbol="SPY",
            quantity=10.0,
            entry_price=100.0,
            entry_time=datetime.now(),
            side="long",
            unrealized_pnl=50.0
        )
        strategy.state.open_positions["SPY"] = position
        strategy.state.daily_trades = 5
        strategy.state.daily_pnl = 25.0
        strategy.state.total_pnl = 200.0
        
        status = strategy.get_status()
        
        expected_keys = {
            'is_active', 'open_positions', 'daily_trades', 'daily_pnl',
            'total_pnl', 'symbols_tracked', 'position_details'
        }
        assert set(status.keys()) == expected_keys
        
        assert status['open_positions'] == 1
        assert status['daily_trades'] == 5
        assert status['daily_pnl'] == 25.0
        assert status['total_pnl'] == 200.0
        assert "SPY" in status['position_details']
        assert status['position_details']['SPY']['side'] == 'long'
        assert status['position_details']['SPY']['unrealized_pnl'] == 50.0


class TestMomentumStrategy:
    """Test MomentumStrategy implementation."""
    
    @pytest.fixture
    def momentum_config(self):
        """Create momentum strategy configuration."""
        return StrategyConfig(
            symbols=["SPY", "AAPL"],
            buy_threshold=0.6,
            sell_threshold=0.6
        )
    
    @pytest.fixture
    def momentum_strategy(self, momentum_config):
        """Create momentum strategy instance."""
        return MomentumStrategy(momentum_config)
    
    def test_momentum_strategy_initialization(self, momentum_strategy, momentum_config):
        """Test momentum strategy initialization."""
        assert isinstance(momentum_strategy, BaseStrategy)
        assert momentum_strategy.config == momentum_config
    
    def test_generate_signals_basic(self, momentum_strategy):
        """Test basic signal generation."""
        model_predictions = {
            "SPY": {"direction": "up", "confidence": 0.8},
            "AAPL": {"direction": "down", "confidence": 0.7}
        }
        current_prices = {"SPY": 100.0, "AAPL": 150.0}
        
        signals = momentum_strategy.generate_signals(model_predictions, current_prices)
        
        assert "SPY" in signals
        assert "AAPL" in signals
        
        # Check SPY signal (up prediction)
        spy_signal = signals["SPY"]
        assert spy_signal['action'] == 'buy'
        assert spy_signal['confidence'] == 0.8
        
        # Check AAPL signal (down prediction)
        aapl_signal = signals["AAPL"]
        assert aapl_signal['action'] == 'sell'
        assert aapl_signal['confidence'] == 0.7
    
    def test_generate_signals_hold(self, momentum_strategy):
        """Test signal generation for hold direction."""
        model_predictions = {
            "SPY": {"direction": "hold", "confidence": 0.5}
        }
        current_prices = {"SPY": 100.0}
        
        signals = momentum_strategy.generate_signals(model_predictions, current_prices)
        
        spy_signal = signals["SPY"]
        assert spy_signal['action'] == 'hold'
    
    def test_generate_signals_missing_predictions(self, momentum_strategy):
        """Test signal generation when predictions are missing."""
        model_predictions = {}  # No predictions
        current_prices = {"SPY": 100.0, "AAPL": 150.0}
        
        signals = momentum_strategy.generate_signals(model_predictions, current_prices)
        
        # Should generate hold signals for all symbols
        assert signals["SPY"]['action'] == 'hold'
        assert signals["AAPL"]['action'] == 'hold'
        assert signals["SPY"]['confidence'] == 0.0
        assert signals["AAPL"]['confidence'] == 0.0
    
    def test_generate_signals_structure(self, momentum_strategy):
        """Test signal structure and metadata."""
        model_predictions = {
            "SPY": {"direction": "up", "confidence": 0.8}
        }
        current_prices = {"SPY": 100.0}
        
        signals = momentum_strategy.generate_signals(model_predictions, current_prices)
        
        spy_signal = signals["SPY"]
        expected_keys = {'action', 'confidence', 'size', 'metadata'}
        assert set(spy_signal.keys()) == expected_keys
        
        # Check metadata
        metadata = spy_signal['metadata']
        assert metadata['strategy'] == 'momentum'
        assert 'raw_prediction' in metadata
        assert 'current_price' in metadata
    
    def test_apply_momentum_filter(self, momentum_strategy):
        """Test momentum filter application."""
        predictions = {"direction": "up", "confidence": 0.8}
        base_confidence = 0.8
        
        filtered_confidence = momentum_strategy._apply_momentum_filter(
            "SPY", predictions, base_confidence
        )
        
        # Currently just returns base confidence
        assert filtered_confidence == base_confidence


class TestCreateStrategy:
    """Test create_strategy factory function."""
    
    def test_create_momentum_strategy(self):
        """Test creating momentum strategy."""
        config = StrategyConfig(symbols=["SPY"])
        strategy = create_strategy("momentum", config)
        
        assert isinstance(strategy, MomentumStrategy)
        assert strategy.config == config
    
    def test_create_strategy_default_config(self):
        """Test creating strategy with default config."""
        strategy = create_strategy("momentum")
        
        assert isinstance(strategy, MomentumStrategy)
        assert isinstance(strategy.config, StrategyConfig)
    
    def test_create_unknown_strategy(self):
        """Test creating unknown strategy type."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy("unknown_strategy")


class TestStrategyIntegration:
    """Integration tests for strategy framework."""
    
    def test_complete_trading_cycle(self):
        """Test complete trading cycle with momentum strategy."""
        config = StrategyConfig(
            symbols=["SPY"],
            buy_threshold=0.6,
            sell_threshold=0.6,
            max_daily_trades=10
        )
        strategy = MomentumStrategy(config)
        strategy.start()
        
        # Generate buy signal
        predictions = {"SPY": {"direction": "up", "confidence": 0.8}}
        current_prices = {"SPY": 100.0}
        
        signals = strategy.generate_signals(predictions, current_prices)
        orders = strategy.process_signals(signals)
        
        # Should create buy order
        assert len(orders) == 1
        assert orders[0]['side'] == 'buy'
        assert "SPY" in strategy.state.open_positions
        
        # Update position with new price
        strategy.update_positions({"SPY": 105.0})
        assert strategy.state.open_positions["SPY"].unrealized_pnl == 50.0
        
        # Generate sell signal
        predictions = {"SPY": {"direction": "down", "confidence": 0.8}}
        current_prices = {"SPY": 105.0}
        
        signals = strategy.generate_signals(predictions, current_prices)
        orders = strategy.process_signals(signals)
        
        # Should close position
        assert len(orders) == 1
        assert orders[0]['side'] == 'sell'
        assert "SPY" not in strategy.state.open_positions
        assert strategy.state.total_pnl == 50.0
    
    def test_risk_management_integration(self):
        """Test risk management features."""
        config = StrategyConfig(
            symbols=["SPY", "AAPL"],
            max_daily_trades=2,  # Low limit for testing
            max_open_positions=1,
            daily_loss_limit=100.0
        )
        strategy = MomentumStrategy(config)
        strategy.start()
        
        # First trade should succeed
        predictions = {"SPY": {"direction": "up", "confidence": 0.8}}
        current_prices = {"SPY": 100.0}
        
        signals = strategy.generate_signals(predictions, current_prices)
        orders = strategy.process_signals(signals)
        assert len(orders) == 1
        
        # Second symbol should be rejected due to position limit
        predictions = {"AAPL": {"direction": "up", "confidence": 0.8}}
        current_prices = {"AAPL": 150.0}
        
        signals = strategy.generate_signals(predictions, current_prices)
        orders = strategy.process_signals(signals)
        assert len(orders) == 0  # Rejected due to position limit
        
        # Close first position
        strategy.close_position("SPY", 95.0)  # At a loss
        
        # Now second position, but should be rejected due to daily loss limit
        if strategy.state.daily_pnl <= -config.daily_loss_limit:
            signals = strategy.generate_signals(predictions, current_prices)
            orders = strategy.process_signals(signals)
            assert len(orders) == 0  # Rejected due to loss limit


if __name__ == "__main__":
    pytest.main([__file__])