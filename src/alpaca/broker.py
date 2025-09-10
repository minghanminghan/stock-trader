#!/usr/bin/env python3
"""
Simple Alpaca Trading Broker Wrapper

Provides a clean interface for order placement and portfolio tracking.
Supports single and batch orders with proper error handling.
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.trading.models import Order

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from src.utils.logging_config import logger


@dataclass
class OrderRequest:
    """Simple order request structure."""
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'
    order_type: str = 'market'  # 'market' or 'limit'
    limit_price: Optional[float] = None
    time_in_force: str = 'day'  # 'day', 'gtc', 'ioc', 'fok'


@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    qty: Optional[float] = None
    side: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class AlpacaBroker:
    """
    Simple wrapper around Alpaca Trading API.
    
    Provides methods for:
    - Single and batch order placement
    - Portfolio tracking and account info
    - Order status monitoring
    - Position management
    """
    
    def __init__(self, paper: bool = True):
        """
        Initialize broker connection.
        
        Args:
            paper: Use paper trading environment (default: True)
        """
        self.paper = paper
        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=paper
        )
        
        # Cache for account and position data
        self._account_cache = None
        self._positions_cache = None
        self._cache_timestamp = None
        self.cache_ttl = 30  # seconds
        
        logger.info(f"AlpacaBroker initialized - Paper trading: {paper}")
    
    def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Place a single order.
        
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
            
            # Create Alpaca order request
            alpaca_request = self._create_alpaca_request(order)
            
            # Submit order
            response: Order = self.client.submit_order(alpaca_request)
            
            logger.info(f"Order placed: {order.side.upper()} {order.qty} {order.symbol} - ID: {response.id}")
            
            return OrderResult(
                success=True,
                order_id=str(response.id),
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                status=response.status.value
            )
            
        except APIError as e:
            error_msg = f"Alpaca API error: {e}"
            logger.error(error_msg)
            return OrderResult(success=False, error=error_msg)
            
        except Exception as e:
            error_msg = f"Order placement error: {e}"
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
            
            # Brief pause between orders to avoid rate limits
            if len(orders) > 1:
                import time
                time.sleep(0.1)
        
        successful_orders = sum(1 for r in results if r.success)
        logger.info(f"Batch order execution: {successful_orders}/{len(orders)} successful")
        
        return results
    
    def get_account(self, use_cache: bool = True) -> Optional[Dict]:
        """
        Get account information.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Account information dictionary
        """
        if use_cache and self._is_cache_valid():
            return self._account_cache
        
        try:
            account = self.client.get_account()
            
            account_data = {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': bool(account.pattern_day_trader),
                'trading_blocked': bool(account.trading_blocked),
                'account_blocked': bool(account.account_blocked),
                'last_updated': datetime.now()
            }
            
            self._account_cache = account_data
            self._cache_timestamp = datetime.now()
            
            return account_data
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return None
    
    def get_positions(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get current positions.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with position information
        """
        if use_cache and self._is_cache_valid() and self._positions_cache is not None:
            return self._positions_cache.copy()
        
        try:
            positions = self.client.get_all_positions()
            
            if not positions:
                empty_df = pd.DataFrame(columns=['symbol', 'qty', 'side', 'market_value', 'cost_basis', 'unrealized_pl', 'unrealized_plpc'])
                self._positions_cache = empty_df
                return empty_df
            
            positions_data = []
            for pos in positions:
                positions_data.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': 'long' if float(pos.qty) > 0 else 'short',
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'avg_entry_price': float(pos.avg_entry_price)
                })
            
            df = pd.DataFrame(positions_data)
            self._positions_cache = df
            self._cache_timestamp = datetime.now()
            
            return df.copy()
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return pd.DataFrame()
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position information or None if no position
        """
        positions = self.get_positions()
        
        if positions.empty:
            return None
        
        symbol_pos = positions[positions['symbol'] == symbol]
        
        if symbol_pos.empty:
            return None
        
        return symbol_pos.iloc[0].to_dict()
    
    def get_buying_power(self) -> Optional[float]:
        """Get available buying power."""
        account = self.get_account()
        return account['buying_power'] if account else None
    
    def get_portfolio_value(self) -> Optional[float]:
        """Get total portfolio value."""
        account = self.get_account()
        return account['portfolio_value'] if account else None
    
    def can_place_order(self, order: OrderRequest) -> bool:
        """
        Check if order can be placed based on buying power and positions.
        
        Args:
            order: OrderRequest to validate
            
        Returns:
            True if order can be placed
        """
        # Get current buying power
        buying_power = self.get_buying_power()
        if buying_power is None:
            return False
        
        # For buy orders, check buying power
        if order.side.lower() == 'buy':
            # Rough estimate - would need current price for exact calculation
            estimated_cost = order.qty * 100  # Assume $100 per share as rough estimate
            return buying_power >= estimated_cost
        
        # For sell orders, check if we have the position
        elif order.side.lower() == 'sell':
            position = self.get_position(order.symbol)
            if position is None:
                return False
            return abs(position['qty']) >= order.qty
        
        return False
    
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
    
    def _validate_order(self, order: OrderRequest) -> bool:
        """Validate order parameters."""
        if not order.symbol or not isinstance(order.symbol, str):
            return False
        
        if order.qty <= 0:
            return False
        
        if order.side.lower() not in ['buy', 'sell']:
            return False
        
        if order.order_type.lower() not in ['market', 'limit']:
            return False
        
        if order.order_type.lower() == 'limit' and order.limit_price is None:
            return False
        
        return True
    
    def _create_alpaca_request(self, order: OrderRequest):
        """Convert OrderRequest to Alpaca API request."""
        side = OrderSide.BUY if order.side.lower() == 'buy' else OrderSide.SELL
        
        # Convert time in force
        tif_mapping = {
            'day': TimeInForce.DAY,
            'gtc': TimeInForce.GTC,
            'ioc': TimeInForce.IOC,
            'fok': TimeInForce.FOK
        }
        time_in_force = tif_mapping.get(order.time_in_force.lower(), TimeInForce.DAY)
        
        # For now, only support market orders to keep it simple
        return MarketOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            side=side,
            time_in_force=time_in_force
        )
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._cache_timestamp is None:
            return False
        
        return (datetime.now() - self._cache_timestamp).total_seconds() < self.cache_ttl


# Convenience functions for easy testing
def create_buy_order(symbol: str, qty: float) -> OrderRequest:
    """Create a simple buy market order."""
    return OrderRequest(symbol=symbol, qty=qty, side='buy')


def create_sell_order(symbol: str, qty: float) -> OrderRequest:
    """Create a simple sell market order."""
    return OrderRequest(symbol=symbol, qty=qty, side='sell')


if __name__ == "__main__":
    # Example usage
    broker = AlpacaBroker(paper=True)
    
    # Get account info
    account = broker.get_account()
    if account:
        print(f"Portfolio Value: ${account['portfolio_value']:.2f}")
        print(f"Buying Power: ${account['buying_power']:.2f}")
    
    # Get positions
    positions = broker.get_positions()
    print(f"Current positions: {len(positions)}")
    print(positions)
    
    # Example order (commented out for safety)
    # buy_order = create_buy_order("SPY", 1)
    # result = broker.place_order(buy_order)
    # print(f"Order result: {result}")