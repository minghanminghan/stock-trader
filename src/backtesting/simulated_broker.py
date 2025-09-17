#!/usr/bin/env python3
"""
Simulated Broker for Backtesting

Provides the same interface as AlpacaBroker but simulates trading without real money.
Tracks portfolio state, executes orders at historical prices, and calculates performance.

Features:
- Complete AlpacaBroker interface compatibility
- Realistic order execution with slippage and fees
- Portfolio tracking and performance calculation
- Position management and buying power constraints
- Trade history and analytics
"""

import uuid
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from enum import Enum

from src.alpaca.broker import OrderRequest, OrderResult
from src.utils.logging_config import logger


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    """Individual trade record."""
    id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fees: float
    order_id: str


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    quantity: float
    avg_entry_price: float
    total_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def unrealized_pl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.total_cost

    @property
    def unrealized_plpc(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.total_cost == 0:
            return 0.0
        return (self.unrealized_pl / abs(self.total_cost)) * 100

    @property
    def side(self) -> str:
        """Position side (long/short)."""
        return 'long' if self.quantity > 0 else 'short' if self.quantity < 0 else 'flat'


@dataclass
class SimulatedAccount:
    """Simulated trading account."""
    initial_capital: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    fees_paid: float = 0.0

    @property
    def equity(self) -> float:
        """Total account equity."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (same as equity)."""
        return self.equity

    @property
    def buying_power(self) -> float:
        """Available buying power (simplified - just cash)."""
        return self.cash

    @property
    def total_return(self) -> float:
        """Total return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100

    @property
    def unrealized_pl(self) -> float:
        """Total unrealized P&L."""
        return sum(pos.unrealized_pl for pos in self.positions.values())


class SimulatedBroker:
    """
    Simulated broker that mimics AlpacaBroker interface for backtesting.

    Executes trades using historical prices and tracks portfolio performance.
    Includes realistic trading costs and slippage simulation.
    """

    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission_per_trade: float = 0.0,  # Commission per trade
                 commission_per_share: float = 0.005,  # Commission per share
                 slippage_bps: float = 1.0,  # Slippage in basis points
                 min_commission: float = 1.0):  # Minimum commission per trade
        """
        Initialize simulated broker.

        Args:
            initial_capital: Starting cash amount
            commission_per_trade: Fixed commission per trade
            commission_per_share: Commission per share traded
            slippage_bps: Slippage in basis points (1 bp = 0.01%)
            min_commission: Minimum commission per trade
        """
        self.account = SimulatedAccount(
            initial_capital=initial_capital,
            cash=initial_capital
        )

        # Trading costs
        self.commission_per_trade = commission_per_trade
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps
        self.min_commission = min_commission

        # Current market prices (updated by data stream)
        self.current_prices: Dict[str, float] = {}
        self.current_timestamp: Optional[datetime] = None

        # Order tracking
        self.pending_orders: List[Dict] = []
        self.filled_orders: List[Dict] = []

        # Performance tracking
        self.daily_values: List[Dict] = []

        logger.info(f"SimulatedBroker initialized with ${initial_capital:,.2f} starting capital")

    def update_market_prices(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """
        Update current market prices and timestamp.

        Args:
            prices: Dictionary of symbol -> current price
            timestamp: Current market timestamp
        """
        self.current_prices.update(prices)
        self.current_timestamp = timestamp

        # Update position market values
        for position in self.account.positions.values():
            if position.symbol in prices:
                position.current_price = prices[position.symbol]

        # Record daily portfolio value for performance tracking
        self._record_portfolio_value()

    def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Simulate order execution.

        Args:
            order: OrderRequest object

        Returns:
            OrderResult with execution details
        """
        try:
            # Validate order
            if not self._validate_order(order):
                return OrderResult(
                    success=False,
                    error="Invalid order parameters"
                )

            # Check if we can execute the order
            if not self._can_execute_order(order):
                return OrderResult(
                    success=False,
                    error="Insufficient buying power or position"
                )

            # Get execution price with slippage
            execution_price = self._get_execution_price(order)
            if execution_price is None:
                return OrderResult(
                    success=False,
                    error=f"No market price available for {order.symbol}"
                )

            # Calculate fees
            fees = self._calculate_fees(order.qty)

            # Execute the trade
            order_id = str(uuid.uuid4())
            trade = Trade(
                id=str(uuid.uuid4()),
                timestamp=self.current_timestamp or datetime.now(timezone.utc),
                symbol=order.symbol,
                side=order.side,
                quantity=order.qty,
                price=execution_price,
                fees=fees,
                order_id=order_id
            )

            # Update portfolio
            self._execute_trade(trade)

            logger.info(f"Simulated order executed: {order.side.upper()} {order.qty} {order.symbol} @ ${execution_price:.2f}")

            return OrderResult(
                success=True,
                order_id=order_id,
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                status="filled"
            )

        except Exception as e:
            error_msg = f"Simulated order execution error: {e}"
            logger.error(error_msg)
            return OrderResult(success=False, error=error_msg)

    def place_orders(self, orders: List[OrderRequest]) -> List[OrderResult]:
        """
        Place multiple orders.

        Args:
            orders: List of OrderRequest objects

        Returns:
            List of OrderResult objects
        """
        results = []
        for order in orders:
            result = self.place_order(order)
            results.append(result)

        successful_orders = sum(1 for r in results if r.success)
        logger.info(f"Simulated batch execution: {successful_orders}/{len(orders)} successful")

        return results

    def get_account(self, use_cache: bool = True) -> Optional[Dict]:
        """
        Get account information.

        Args:
            use_cache: Ignored for simulated broker

        Returns:
            Account information dictionary
        """
        try:
            return {
                'equity': self.account.equity,
                'cash': self.account.cash,
                'buying_power': self.account.buying_power,
                'portfolio_value': self.account.portfolio_value,
                'pattern_day_trader': False,  # Simplified
                'trading_blocked': False,
                'account_blocked': False,
                'last_updated': self.current_timestamp or datetime.now(timezone.utc),
                'total_return_pct': self.account.total_return,
                'unrealized_pl': self.account.unrealized_pl,
                'fees_paid': self.account.fees_paid
            }
        except Exception as e:
            logger.error(f"Error getting simulated account info: {e}")
            return None

    def get_positions(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get current positions.

        Args:
            use_cache: Ignored for simulated broker

        Returns:
            DataFrame with position information
        """
        try:
            if not self.account.positions:
                return pd.DataFrame(columns=[
                    'symbol', 'qty', 'side', 'market_value', 'cost_basis',
                    'unrealized_pl', 'unrealized_plpc', 'avg_entry_price'
                ])

            positions_data = []
            for position in self.account.positions.values():
                if position.quantity != 0:  # Only show non-zero positions
                    positions_data.append({
                        'symbol': position.symbol,
                        'qty': position.quantity,
                        'side': position.side,
                        'market_value': position.market_value,
                        'cost_basis': position.total_cost,
                        'unrealized_pl': position.unrealized_pl,
                        'unrealized_plpc': position.unrealized_plpc,
                        'avg_entry_price': position.avg_entry_price
                    })

            return pd.DataFrame(positions_data)

        except Exception as e:
            logger.error(f"Error getting simulated positions: {e}")
            return pd.DataFrame()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position information or None if no position
        """
        if symbol not in self.account.positions:
            return None

        position = self.account.positions[symbol]
        if position.quantity == 0:
            return None

        return {
            'symbol': position.symbol,
            'qty': position.quantity,
            'side': position.side,
            'market_value': position.market_value,
            'cost_basis': position.total_cost,
            'unrealized_pl': position.unrealized_pl,
            'unrealized_plpc': position.unrealized_plpc,
            'avg_entry_price': position.avg_entry_price
        }

    def get_buying_power(self) -> Optional[float]:
        """Get available buying power."""
        return self.account.buying_power

    def get_portfolio_value(self) -> Optional[float]:
        """Get total portfolio value."""
        return self.account.portfolio_value

    def can_place_order(self, order: OrderRequest) -> bool:
        """
        Check if order can be placed.

        Args:
            order: OrderRequest to validate

        Returns:
            True if order can be placed
        """
        return self._can_execute_order(order)

    def close_position(self, symbol: str) -> OrderResult:
        """
        Close entire position for a symbol.

        Args:
            symbol: Stock symbol to close

        Returns:
            OrderResult
        """
        position = self.get_position(symbol)

        if position is None:
            return OrderResult(
                success=False,
                error=f"No position found for {symbol}"
            )

        # Determine order side to close position
        qty = abs(position['qty'])
        side = 'sell' if position['qty'] > 0 else 'buy'

        close_order = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=side
        )

        return self.place_order(close_order)

    def get_trade_history(self) -> pd.DataFrame:
        """Get complete trade history."""
        if not self.account.trades:
            return pd.DataFrame(columns=[
                'timestamp', 'symbol', 'side', 'quantity', 'price', 'fees', 'total_value'
            ])

        trades_data = []
        for trade in self.account.trades:
            total_value = trade.quantity * trade.price
            if trade.side == 'sell':
                total_value = -total_value

            trades_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'fees': trade.fees,
                'total_value': total_value
            })

        return pd.DataFrame(trades_data)

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        trades_df = self.get_trade_history()

        if trades_df.empty:
            return {
                'total_trades': 0,
                'total_return_pct': 0.0,
                'total_return_dollar': 0.0,
                'total_fees': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

        # Calculate basic metrics
        total_trades = len(trades_df)
        total_return_dollar = self.account.portfolio_value - self.account.initial_capital
        total_return_pct = self.account.total_return

        # Calculate trade-level profits for win rate
        trade_profits = []
        if total_trades > 0:
            # Group trades by symbol to calculate round-trip P&L
            for symbol in trades_df['symbol'].unique():
                symbol_trades = trades_df[trades_df['symbol'] == symbol].sort_values('timestamp')
                position = 0
                avg_price = 0

                for _, trade in symbol_trades.iterrows():
                    if trade['side'] == 'buy':
                        if position <= 0:  # Opening or closing short
                            if position < 0:  # Closing short position
                                profit = (-position) * (avg_price - trade['price']) - trade['fees']
                                trade_profits.append(profit)
                            # Update position
                            if position == 0:
                                avg_price = trade['price']
                            position += trade['quantity']
                    else:  # sell
                        if position >= 0:  # Closing long or opening short
                            if position > 0:  # Closing long position
                                profit = position * (trade['price'] - avg_price) - trade['fees']
                                trade_profits.append(profit)
                            # Update position
                            if position == 0:
                                avg_price = trade['price']
                            position -= trade['quantity']

        # Calculate win rate
        win_rate = 0.0
        profit_factor = 0.0
        if trade_profits:
            winning_trades = [p for p in trade_profits if p > 0]
            losing_trades = [p for p in trade_profits if p < 0]
            win_rate = len(winning_trades) / len(trade_profits) * 100

            if losing_trades:
                profit_factor = sum(winning_trades) / abs(sum(losing_trades))

        # Calculate Sharpe ratio and max drawdown from daily values
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()

        return {
            'total_trades': total_trades,
            'total_return_pct': total_return_pct,
            'total_return_dollar': total_return_dollar,
            'total_fees': self.account.fees_paid,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_positions': len([p for p in self.account.positions.values() if p.quantity != 0])
        }

    def _validate_order(self, order: OrderRequest) -> bool:
        """Validate order parameters."""
        if not order.symbol or not isinstance(order.symbol, str):
            return False
        if order.qty <= 0:
            return False
        if order.side.lower() not in ['buy', 'sell']:
            return False
        return True

    def _can_execute_order(self, order: OrderRequest) -> bool:
        """Check if order can be executed."""
        if order.side.lower() == 'buy':
            # Check buying power
            estimated_cost = order.qty * self.current_prices.get(order.symbol, 100)  # Default price
            fees = self._calculate_fees(order.qty)
            return self.account.cash >= (estimated_cost + fees)

        elif order.side.lower() == 'sell':
            # Check if we have enough shares
            position = self.account.positions.get(order.symbol)
            if position is None:
                return False
            return position.quantity >= order.qty

        return False

    def _get_execution_price(self, order: OrderRequest) -> Optional[float]:
        """Get execution price with slippage."""
        if order.symbol not in self.current_prices:
            return None

        base_price = self.current_prices[order.symbol]

        # Apply slippage
        slippage_factor = self.slippage_bps / 10000  # Convert basis points to decimal

        if order.side.lower() == 'buy':
            # Buy orders get worse (higher) price
            execution_price = base_price * (1 + slippage_factor)
        else:
            # Sell orders get worse (lower) price
            execution_price = base_price * (1 - slippage_factor)

        return execution_price

    def _calculate_fees(self, quantity: float) -> float:
        """Calculate trading fees."""
        fees = self.commission_per_trade + (quantity * self.commission_per_share)
        return max(fees, self.min_commission)

    def _execute_trade(self, trade: Trade) -> None:
        """Execute trade and update portfolio."""
        # Add trade to history
        self.account.trades.append(trade)

        # Update cash
        if trade.side == 'buy':
            cash_change = -(trade.quantity * trade.price + trade.fees)
        else:  # sell
            cash_change = trade.quantity * trade.price - trade.fees

        self.account.cash += cash_change
        self.account.fees_paid += trade.fees

        # Update position
        if trade.symbol not in self.account.positions:
            self.account.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=0,
                avg_entry_price=0,
                total_cost=0,
                current_price=self.current_prices.get(trade.symbol, trade.price)
            )

        position = self.account.positions[trade.symbol]

        if trade.side == 'buy':
            # Calculate new average entry price
            total_shares = position.quantity + trade.quantity
            if total_shares > 0:
                total_cost = position.total_cost + (trade.quantity * trade.price)
                position.avg_entry_price = total_cost / total_shares
                position.total_cost = total_cost
            position.quantity += trade.quantity
        else:  # sell
            # Reduce position
            position.quantity -= trade.quantity
            if position.quantity == 0:
                position.avg_entry_price = 0
                position.total_cost = 0
            else:
                # Proportionally reduce total cost
                remaining_fraction = position.quantity / (position.quantity + trade.quantity)
                position.total_cost *= remaining_fraction

    def _record_portfolio_value(self) -> None:
        """Record daily portfolio value for performance tracking."""
        if self.current_timestamp:
            self.daily_values.append({
                'timestamp': self.current_timestamp,
                'portfolio_value': self.account.portfolio_value,
                'cash': self.account.cash,
                'equity': self.account.equity
            })

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if len(self.daily_values) < 2:
            return 0.0

        # Calculate daily returns
        values = [d['portfolio_value'] for d in self.daily_values]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualized Sharpe ratio (assuming 252 trading days)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0

        return sharpe

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.daily_values) < 2:
            return 0.0

        values = [d['portfolio_value'] for d in self.daily_values]

        # Calculate running maximum and drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max * 100

        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0


if __name__ == "__main__":
    """Test the simulated broker."""

    # Create simulated broker
    broker = SimulatedBroker(initial_capital=100000)

    # Simulate some market prices
    broker.update_market_prices({
        'AAPL': 150.0,
        'MSFT': 300.0,
        'NVDA': 800.0
    }, datetime.now(timezone.utc))

    # Test account info
    account = broker.get_account()
    print(f"Initial portfolio value: ${account['portfolio_value']:,.2f}")

    # Test buy order
    from src.alpaca.broker import OrderRequest
    buy_order = OrderRequest(symbol='AAPL', qty=10, side='buy')
    result = broker.place_order(buy_order)
    print(f"Buy order result: {result}")

    # Test sell order
    sell_order = OrderRequest(symbol='AAPL', qty=5, side='sell')
    result = broker.place_order(sell_order)
    print(f"Sell order result: {result}")

    # Check positions
    positions = broker.get_positions()
    print(f"Positions:\n{positions}")

    # Check performance
    performance = broker.get_performance_summary()
    print(f"Performance summary: {performance}")