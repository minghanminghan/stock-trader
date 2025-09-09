from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.trading.portfolio import Portfolio
from src.trading.strategy import TradingDecision, Action
from src.utils.logging_config import logger


class RiskManager:
    """
    Risk management and position sizing for trading strategy.
    
    Controls:
    - Position sizing based on volatility and account equity
    - Stop loss and take profit levels  
    - Daily loss limits and circuit breakers
    - Portfolio concentration limits
    """
    
    def __init__(self, 
                 max_position_risk: float = 0.02,  # 2% max risk per position
                 daily_loss_limit: float = 0.03,   # 3% daily loss limit
                 max_portfolio_positions: int = 10,
                 stop_loss_pct: float = 0.005,     # 0.5% stop loss
                 take_profit_pct: float = 0.01,    # 1% take profit
                 max_symbol_concentration: float = 0.25):  # 25% max per symbol
        """
        Initialize risk manager.
        
        Args:
            max_position_risk: Max % of equity to risk per position
            daily_loss_limit: Max % daily loss before stopping trading
            max_portfolio_positions: Max number of open positions
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_symbol_concentration: Max % of portfolio in single symbol
        """
        self.max_position_risk = max_position_risk
        self.daily_loss_limit = daily_loss_limit
        self.max_portfolio_positions = max_portfolio_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_symbol_concentration = max_symbol_concentration
        
        # Circuit breaker state
        self.trading_halted = False
        self.halt_reason = ""
        self.daily_start_equity = 0.0
        
        # Volatility cache for ATR calculations
        self.atr_cache: Dict[str, float] = {}
        self.price_history: Dict[str, list] = {}
        
        logger.info("Risk manager initialized")
    
    def set_daily_start_equity(self, equity: float) -> None:
        """Set starting equity for daily loss calculations."""
        self.daily_start_equity = equity
        self.trading_halted = False
        self.halt_reason = ""
        logger.info(f"Daily start equity set: ${equity:,.2f}")
    
    def calculate_position_size(self, symbol: str, decision: TradingDecision, 
                               account_equity: float, current_price: float,
                               portfolio: Portfolio) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Stock symbol
            decision: Trading decision from strategy
            account_equity: Current account equity
            current_price: Current stock price
            portfolio: Current portfolio state
            
        Returns:
            Adjusted position size (shares)
        """
        if self.trading_halted:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return 0
        
        if decision.action == Action.HOLD:
            return 0
        
        # Get ATR for volatility-based sizing
        atr = self._get_atr(symbol, current_price)
        
        # Calculate base position size using ATR
        # Position size = (equity * risk_per_trade) / (ATR * 2)  # 2x ATR as stop distance
        max_risk_dollars = account_equity * self.max_position_risk
        stop_distance = max(atr * 2, current_price * self.stop_loss_pct)
        
        base_size = int(max_risk_dollars / stop_distance)
        
        # Adjust for confidence
        confidence_adjusted_size = int(base_size * decision.confidence)
        
        # Apply concentration limits
        max_concentration_size = self._calculate_concentration_limit(
            symbol, account_equity, current_price, portfolio
        )
        
        # Apply position count limits
        if decision.action in [Action.BUY, Action.SELL]:
            if portfolio.get_position_count() >= self.max_portfolio_positions:
                existing_pos = portfolio.get_position(symbol)
                if existing_pos is None:  # New position would exceed limit
                    logger.warning(f"Max positions reached: {self.max_portfolio_positions}")
                    return 0
        
        # Take minimum of all constraints
        final_size = min(
            confidence_adjusted_size,
            max_concentration_size,
            decision.qty  # Don't exceed strategy's requested size
        )
        
        # Minimum position size check
        min_size = max(1, int(100 / current_price))  # At least $100 position
        final_size = max(final_size, min_size) if final_size > 0 else 0
        
        logger.debug(f"Position sizing {symbol}: base={base_size}, "
                    f"confidence_adj={confidence_adjusted_size}, "
                    f"concentration_limit={max_concentration_size}, final={final_size}")
        
        return final_size
    
    def _get_atr(self, symbol: str, current_price: float, period: int = 14) -> float:
        """
        Calculate Average True Range for volatility estimation.
        Simplified version using price history.
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            period: ATR calculation period
            
        Returns:
            ATR value
        """
        # Add current price to history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(current_price)
        
        # Keep only recent prices
        if len(self.price_history[symbol]) > period * 2:
            self.price_history[symbol] = self.price_history[symbol][-period * 2:]
        
        prices = self.price_history[symbol]
        
        # Need at least 2 prices for ATR
        if len(prices) < 2:
            # Use simple percentage volatility as fallback
            atr = current_price * 0.01  # 1% of price
        else:
            # Simple range-based ATR approximation
            ranges = []
            for i in range(1, len(prices)):
                daily_range = abs(prices[i] - prices[i-1])
                ranges.append(daily_range)
            
            atr = np.mean(ranges[-period:]) if len(ranges) >= period else np.mean(ranges)
            
            # Ensure reasonable ATR bounds
            min_atr = current_price * 0.005  # 0.5% minimum
            max_atr = current_price * 0.05   # 5% maximum
            atr = max(min_atr, min(atr, max_atr))
        
        self.atr_cache[symbol] = atr
        return atr
    
    def _calculate_concentration_limit(self, symbol: str, account_equity: float,
                                     current_price: float, portfolio: Portfolio) -> int:
        """Calculate maximum position size based on concentration limits."""
        max_symbol_value = account_equity * self.max_symbol_concentration
        
        # Account for existing position
        existing_exposure = portfolio.get_exposure(symbol)
        remaining_capacity = max_symbol_value - existing_exposure
        
        if remaining_capacity <= 0:
            return 0
        
        max_shares = int(remaining_capacity / current_price)
        return max_shares
    
    def check_daily_loss_limit(self, current_equity: float) -> bool:
        """
        Check if daily loss limit has been breached.
        
        Args:
            current_equity: Current account equity
            
        Returns:
            True if trading should continue, False if halted
        """
        if self.daily_start_equity == 0:
            self.set_daily_start_equity(current_equity)
            return True
        
        daily_pl = (current_equity - self.daily_start_equity) / self.daily_start_equity
        
        if daily_pl <= -self.daily_loss_limit:
            self.trading_halted = True
            self.halt_reason = f"Daily loss limit breached: {daily_pl:.2%} <= -{self.daily_loss_limit:.2%}"
            logger.error(self.halt_reason)
            return False
        
        return True
    
    def should_stop_out(self, symbol: str, current_price: float, 
                       portfolio: Portfolio) -> bool:
        """
        Check if position should be stopped out due to loss.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            portfolio: Portfolio state
            
        Returns:
            True if position should be closed
        """
        position = portfolio.get_position(symbol)
        if position is None:
            return False
        
        # Calculate loss percentage
        if position.side == 'long':
            loss_pct = (position.avg_entry_price - current_price) / position.avg_entry_price
        else:  # short
            loss_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
        
        # Stop loss trigger
        if loss_pct >= self.stop_loss_pct:
            logger.warning(f"Stop loss triggered for {symbol}: {loss_pct:.3%}")
            return True
        
        return False
    
    def should_take_profit(self, symbol: str, current_price: float,
                          portfolio: Portfolio) -> bool:
        """
        Check if position should be closed for profit taking.
        
        Args:
            symbol: Stock symbol  
            current_price: Current market price
            portfolio: Portfolio state
            
        Returns:
            True if position should be closed
        """
        position = portfolio.get_position(symbol)
        if position is None:
            return False
        
        # Calculate profit percentage
        if position.side == 'long':
            profit_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
        else:  # short
            profit_pct = (position.avg_entry_price - current_price) / position.avg_entry_price
        
        # Take profit trigger
        if profit_pct >= self.take_profit_pct:
            logger.info(f"Take profit triggered for {symbol}: {profit_pct:.3%}")
            return True
        
        return False
    
    def validate_trade(self, symbol: str, decision: TradingDecision,
                      account_equity: float, current_price: float,
                      portfolio: Portfolio) -> Tuple[bool, str]:
        """
        Validate if trade should be executed based on risk rules.
        
        Args:
            symbol: Stock symbol
            decision: Trading decision
            account_equity: Current equity
            current_price: Current price
            portfolio: Portfolio state
            
        Returns:
            (is_valid, reason) tuple
        """
        # Check if trading is halted
        if self.trading_halted:
            return False, f"Trading halted: {self.halt_reason}"
        
        # Check daily loss limit
        if not self.check_daily_loss_limit(account_equity):
            return False, "Daily loss limit breached"
        
        # For hold decisions, always valid
        if decision.action == Action.HOLD:
            return True, "Hold decision"
        
        # Check position limits for new positions
        if decision.action in [Action.BUY, Action.SELL]:
            existing_pos = portfolio.get_position(symbol)
            if existing_pos is None:  # New position
                if portfolio.get_position_count() >= self.max_portfolio_positions:
                    return False, f"Max positions limit reached: {self.max_portfolio_positions}"
        
        # Check concentration limits
        max_size = self._calculate_concentration_limit(symbol, account_equity, current_price, portfolio)
        if max_size <= 0:
            return False, f"Symbol concentration limit exceeded: {self.max_symbol_concentration:.1%}"
        
        # Check minimum trade size
        min_trade_value = 100  # $100 minimum
        trade_value = decision.qty * current_price
        if trade_value < min_trade_value:
            return False, f"Trade size too small: ${trade_value:.2f} < ${min_trade_value}"
        
        return True, "Trade validated"
    
    def get_risk_metrics(self, portfolio: Portfolio, account_equity: float) -> Dict:
        """Get current risk metrics."""
        daily_pl = 0.0
        daily_pl_pct = 0.0
        
        if self.daily_start_equity > 0:
            daily_pl = account_equity - self.daily_start_equity
            daily_pl_pct = daily_pl / self.daily_start_equity
        
        return {
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'daily_pl': daily_pl,
            'daily_pl_pct': daily_pl_pct,
            'daily_loss_limit': self.daily_loss_limit,
            'position_count': portfolio.get_position_count(),
            'max_positions': self.max_portfolio_positions,
            'portfolio_value': portfolio.get_portfolio_value(),
            'account_equity': account_equity
        }