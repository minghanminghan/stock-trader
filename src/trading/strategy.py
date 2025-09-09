from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.trading.portfolio import Portfolio, Position
from src.utils.logging_config import logger


class Signal(Enum):
    """ML model signal types."""
    SELL = 0
    HOLD = 1
    BUY = 2


class Action(Enum):
    """Trading actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingDecision:
    """Structured trading decision."""
    symbol: str
    action: Action
    qty: int
    reason: str
    confidence: float = 0.0


@dataclass
class SignalData:
    """ML signal with metadata."""
    symbol: str
    prediction: Signal
    confidence: float
    timestamp: str


class MomentumTradingStrategy:
    """
    Momentum-based intraday trading strategy.
    
    Rules:
    - Enter Long: if signal = BUY and no existing long position
    - Enter Short: if signal = SELL and no existing short position  
    - Hold: if signal = HOLD, maintain current position
    - Exit: if signal flips against position
    """
    
    def __init__(self, enable_short_selling: bool = True, 
                 min_confidence: float = 0.6,
                 hold_threshold: float = 0.05):
        """
        Initialize strategy.
        
        Args:
            enable_short_selling: Whether to allow short positions
            min_confidence: Minimum confidence to act on signals
            hold_threshold: Confidence threshold for holding vs acting
        """
        self.enable_short_selling = enable_short_selling
        self.min_confidence = min_confidence
        self.hold_threshold = hold_threshold
        
        logger.info(f"Momentum strategy initialized - Short selling: {enable_short_selling}")
    
    def evaluate_symbol(self, signal_data: SignalData, 
                       portfolio: Portfolio) -> TradingDecision:
        """
        Evaluate trading decision for a single symbol.
        
        Args:
            signal_data: ML signal and confidence
            portfolio: Current portfolio state
            
        Returns:
            TradingDecision object
        """
        symbol = signal_data.symbol
        signal = signal_data.prediction
        confidence = signal_data.confidence
        
        current_position = portfolio.get_position(symbol)
        
        # Skip if confidence too low
        if confidence < self.min_confidence:
            return TradingDecision(
                symbol=symbol,
                action=Action.HOLD,
                qty=0,
                reason=f"Low confidence: {confidence:.2f} < {self.min_confidence}",
                confidence=confidence
            )
        
        # No existing position - consider entry
        if current_position is None:
            return self._evaluate_entry(signal_data)
        
        # Has existing position - consider exit or hold
        return self._evaluate_exit_or_hold(signal_data, current_position)
    
    def _evaluate_entry(self, signal_data: SignalData) -> TradingDecision:
        """Evaluate entry decision when no position exists."""
        symbol = signal_data.symbol
        signal = signal_data.prediction
        confidence = signal_data.confidence
        
        if signal == Signal.BUY:
            return TradingDecision(
                symbol=symbol,
                action=Action.BUY,
                qty=100,  # Base quantity - will be adjusted by risk manager
                reason=f"Enter long on BUY signal (confidence: {confidence:.2f})",
                confidence=confidence
            )
        
        elif signal == Signal.SELL and self.enable_short_selling:
            return TradingDecision(
                symbol=symbol,
                action=Action.SELL,
                qty=100,  # Base quantity - will be adjusted by risk manager
                reason=f"Enter short on SELL signal (confidence: {confidence:.2f})",
                confidence=confidence
            )
        
        else:
            return TradingDecision(
                symbol=symbol,
                action=Action.HOLD,
                qty=0,
                reason=f"Signal {signal.name} - no entry conditions met",
                confidence=confidence
            )
    
    def _evaluate_exit_or_hold(self, signal_data: SignalData, 
                              position: Position) -> TradingDecision:
        """Evaluate exit/hold decision when position exists."""
        symbol = signal_data.symbol
        signal = signal_data.prediction
        confidence = signal_data.confidence
        
        # Check for signal reversal (exit condition)
        if position.side == 'long':
            if signal == Signal.SELL:
                # Exit long on SELL signal
                return TradingDecision(
                    symbol=symbol,
                    action=Action.SELL,
                    qty=abs(position.qty),
                    reason=f"Exit long on SELL signal (confidence: {confidence:.2f})",
                    confidence=confidence
                )
            elif signal == Signal.HOLD:
                # Check if we should exit on weak signal
                if confidence > (1 - self.hold_threshold):  # High confidence HOLD
                    return TradingDecision(
                        symbol=symbol,
                        action=Action.HOLD,
                        qty=0,
                        reason=f"Hold long position (confidence: {confidence:.2f})",
                        confidence=confidence
                    )
                else:
                    # Low confidence HOLD - consider exiting
                    return TradingDecision(
                        symbol=symbol,
                        action=Action.SELL,
                        qty=abs(position.qty),
                        reason=f"Exit long on weak HOLD signal (confidence: {confidence:.2f})",
                        confidence=confidence
                    )
            else:  # BUY signal
                return TradingDecision(
                    symbol=symbol,
                    action=Action.HOLD,
                    qty=0,
                    reason=f"Hold long on BUY signal (confidence: {confidence:.2f})",
                    confidence=confidence
                )
        
        elif position.side == 'short':
            if signal == Signal.BUY:
                # Cover short on BUY signal
                return TradingDecision(
                    symbol=symbol,
                    action=Action.BUY,
                    qty=abs(position.qty),
                    reason=f"Cover short on BUY signal (confidence: {confidence:.2f})",
                    confidence=confidence
                )
            elif signal == Signal.HOLD:
                # Check if we should cover on weak signal
                if confidence > (1 - self.hold_threshold):  # High confidence HOLD
                    return TradingDecision(
                        symbol=symbol,
                        action=Action.HOLD,
                        qty=0,
                        reason=f"Hold short position (confidence: {confidence:.2f})",
                        confidence=confidence
                    )
                else:
                    # Low confidence HOLD - consider covering
                    return TradingDecision(
                        symbol=symbol,
                        action=Action.BUY,
                        qty=abs(position.qty),
                        reason=f"Cover short on weak HOLD signal (confidence: {confidence:.2f})",
                        confidence=confidence
                    )
            else:  # SELL signal
                return TradingDecision(
                    symbol=symbol,
                    action=Action.HOLD,
                    qty=0,
                    reason=f"Hold short on SELL signal (confidence: {confidence:.2f})",
                    confidence=confidence
                )
        
        # Fallback
        return TradingDecision(
            symbol=symbol,
            action=Action.HOLD,
            qty=0,
            reason="Default hold",
            confidence=confidence
        )
    
    def rank_opportunities(self, signals: List[SignalData], 
                          portfolio: Portfolio) -> List[TradingDecision]:
        """
        Rank trading opportunities across multiple symbols.
        
        Args:
            signals: List of ML signals for different symbols
            portfolio: Current portfolio state
            
        Returns:
            List of trading decisions ranked by attractiveness
        """
        decisions = []
        
        for signal_data in signals:
            decision = self.evaluate_symbol(signal_data, portfolio)
            decisions.append(decision)
        
        # Rank by confidence and action priority
        # Priority: actionable trades > holds
        def decision_score(decision):
            action_weight = {Action.BUY: 2, Action.SELL: 2, Action.HOLD: 0}
            return action_weight[decision.action] * decision.confidence
        
        ranked_decisions = sorted(decisions, key=decision_score, reverse=True)
        
        logger.debug(f"Ranked {len(ranked_decisions)} trading opportunities")
        return ranked_decisions
    
    def should_exit_for_weekend(self, symbol: str, portfolio: Portfolio) -> bool:
        """
        Determine if position should be exited before weekend.
        
        Args:
            symbol: Stock symbol
            portfolio: Current portfolio
            
        Returns:
            True if should exit position
        """
        position = portfolio.get_position(symbol)
        if position is None:
            return False
        
        # Conservative approach - exit all positions before weekend
        # This reduces weekend gap risk
        return True
    
    def get_strategy_stats(self, portfolio: Portfolio) -> Dict:
        """Get strategy performance statistics."""
        return {
            'total_positions': portfolio.get_position_count(),
            'portfolio_value': portfolio.get_portfolio_value(),
            'unrealized_pl': portfolio.get_total_unrealized_pl(),
            'realized_pl_today': portfolio.realized_pl_today,
            'total_trades_today': portfolio.total_trades_today,
            'strategy_type': 'momentum_intraday'
        }