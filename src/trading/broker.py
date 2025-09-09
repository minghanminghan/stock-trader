import asyncio
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import time
import os
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.common.exceptions import APIError

from src.utils.logging_config import logger


class AlpacaBroker:
    """
    Alpaca broker interface for stock trading.
    Handles order submission, position management, and account queries.
    """
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpaca broker.
        
        Args:
            paper_trading: If True, use paper trading environment
        """
        self.paper_trading = paper_trading
        
        # Get API credentials from environment
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper_trading
        )
        
        mode = "PAPER" if paper_trading else "LIVE"
        logger.info(f"AlpacaBroker initialized in {mode} mode")
    
    def submit_order(self, symbol: str, qty: int, side: str, 
                    order_type: str = "market", time_in_force: str = "day",
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> Optional[Dict]:
        """
        Submit order to Alpaca.
        
        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: "buy" or "sell"
            order_type: "market", "limit", or "stop"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order result dict or None if failed
        """
        try:
            # Convert side to Alpaca enum
            alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # Convert time in force
            tif_map = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK
            }
            alpaca_tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
            
            # Create order request based on type
            if order_type.lower() == "market":
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif
                )
            elif order_type.lower() == "limit":
                if limit_price is None:
                    raise ValueError("Limit price required for limit orders")
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price
                )
            elif order_type.lower() == "stop":
                if stop_price is None:
                    raise ValueError("Stop price required for stop orders")
                order_data = StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    stop_price=stop_price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            # Convert to dict format
            result = {
                'id': str(order.id),
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side.value,
                'order_type': order.order_type.value,
                'status': order.status.value,
                'filled_qty': float(order.filled_qty or 0),
                'filled_avg_price': float(order.filled_avg_price or 0),
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                'filled_at': order.filled_at.isoformat() if order.filled_at else None
            }
            
            logger.info(f"Order submitted: {side.upper()} {qty} {symbol} ({order_type}) - ID: {order.id}")
            return result
            
        except APIError as e:
            logger.error(f"Alpaca API error submitting order: {e}")
            return None
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position dict or None if no position
        """
        try:
            position = self.trading_client.get_open_position(symbol)
            
            return {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'side': 'long' if float(position.qty) > 0 else 'short',
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'avg_entry_price': float(position.avg_entry_price)
            }
            
        except APIError as e:
            if "position does not exist" in str(e).lower():
                return None  # No position exists
            logger.error(f"Error getting position for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return None
    
    def get_all_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            List of position dicts
        """
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for position in positions:
                result.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'side': 'long' if float(position.qty) > 0 else 'short',
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'avg_entry_price': float(position.avg_entry_price)
                })
            
            logger.debug(f"Retrieved {len(result)} positions")
            return result
            
        except APIError as e:
            logger.error(f"Error getting all positions: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting all positions: {e}")
            return []
    
    def get_account_balance(self) -> Dict:
        """
        Get account balance and equity information.
        
        Returns:
            Account balance dict
        """
        try:
            account = self.trading_client.get_account()
            
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'long_market_value': float(account.long_market_value or 0),
                'short_market_value': float(account.short_market_value or 0),
                'day_trade_count': int(account.day_trade_count),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked
            }
            
        except APIError as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
    
    def get_orders(self, status: str = "open", limit: int = 50) -> List[Dict]:
        """
        Get orders by status.
        
        Args:
            status: Order status filter ("open", "closed", "all")
            limit: Maximum number of orders to return
            
        Returns:
            List of order dicts
        """
        try:
            from alpaca.trading.enums import QueryOrderStatus
            
            status_map = {
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL
            }
            
            alpaca_status = status_map.get(status.lower(), QueryOrderStatus.OPEN)
            orders = self.trading_client.get_orders(status=alpaca_status, limit=limit)
            
            result = []
            for order in orders:
                result.append({
                    'id': str(order.id),
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': order.side.value,
                    'order_type': order.order_type.value,
                    'status': order.status.value,
                    'filled_qty': float(order.filled_qty or 0),
                    'filled_avg_price': float(order.filled_avg_price or 0),
                    'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None
                })
            
            logger.debug(f"Retrieved {len(result)} {status} orders")
            return result
            
        except APIError as e:
            logger.error(f"Error getting orders: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except APIError as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def close_position(self, symbol: str, qty: Optional[int] = None) -> Optional[Dict]:
        """
        Close position for a symbol.
        
        Args:
            symbol: Stock symbol
            qty: Quantity to close (None for full position)
            
        Returns:
            Order result dict or None if failed
        """
        try:
            # Get current position
            position = self.get_position(symbol)
            if not position:
                logger.warning(f"No position to close for {symbol}")
                return None
            
            current_qty = abs(float(position['qty']))
            close_qty = qty if qty is not None else current_qty
            
            # Determine side (opposite of current position)
            current_side = position['side']
            close_side = "sell" if current_side == "long" else "buy"
            
            # Submit closing order
            return self.submit_order(
                symbol=symbol,
                qty=close_qty,
                side=close_side,
                order_type="market"
            )
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return None


class AsyncAlpacaBroker(AlpacaBroker):
    """
    Async wrapper around AlpacaBroker for non-blocking operations.
    Allows parallel order submission and portfolio queries.
    """
    
    def __init__(self, paper_trading: bool = True, max_workers: int = 4):
        """
        Initialize async broker.
        
        Args:
            paper_trading: If True, use paper trading
            max_workers: Maximum number of concurrent operations
        """
        super().__init__(paper_trading)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"AsyncAlpacaBroker initialized with {max_workers} workers")
    
    async def submit_order_async(self, symbol: str, qty: int, side: str, 
                                order_type: str = "market", time_in_force: str = "day",
                                limit_price: Optional[float] = None,
                                stop_price: Optional[float] = None) -> Optional[Dict]:
        """
        Submit order asynchronously.
        
        Args:
            symbol: Stock symbol
            qty: Quantity 
            side: "buy" or "sell"
            order_type: Order type
            time_in_force: Time in force
            limit_price: Limit price if needed
            stop_price: Stop price if needed
            
        Returns:
            Order result dict or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.submit_order,
            symbol, qty, side, order_type, time_in_force, limit_price, stop_price
        )
    
    async def get_position_async(self, symbol: str) -> Optional[Dict]:
        """Get position asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.get_position, symbol)
    
    async def get_all_positions_async(self) -> List[Dict]:
        """Get all positions asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.get_all_positions)
    
    async def get_account_balance_async(self) -> Dict:
        """Get account balance asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.get_account_balance)
    
    async def submit_multiple_orders_async(self, orders: List[Dict]) -> List[Optional[Dict]]:
        """
        Submit multiple orders concurrently.
        
        Args:
            orders: List of order dicts with keys: symbol, qty, side, etc.
            
        Returns:
            List of order results
        """
        tasks = []
        
        for order in orders:
            task = self.submit_order_async(
                symbol=order['symbol'],
                qty=order['qty'],
                side=order['side'],
                order_type=order.get('order_type', 'market'),
                time_in_force=order.get('time_in_force', 'day'),
                limit_price=order.get('limit_price'),
                stop_price=order.get('stop_price')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Order submission failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_positions_batch_async(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Get positions for multiple symbols concurrently.
        
        Args:
            symbols: List of symbols to query
            
        Returns:
            Dict mapping symbol -> position dict (or None)
        """
        tasks = [self.get_position_async(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        position_map = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Position query failed for {symbol}: {result}")
                position_map[symbol] = None
            else:
                position_map[symbol] = result
        
        return position_map
    
    def close(self):
        """Clean up executor."""
        self.executor.shutdown(wait=True)


class BatchOrderManager:
    """
    Manages batch order operations with rate limiting and retry logic.
    """
    
    def __init__(self, broker: AsyncAlpacaBroker, rate_limit_delay: float = 0.2):
        """
        Initialize batch order manager.
        
        Args:
            broker: Async broker instance
            rate_limit_delay: Delay between order batches (seconds)
        """
        self.broker = broker
        self.rate_limit_delay = rate_limit_delay
        self.pending_orders = []
        
        logger.info("BatchOrderManager initialized")
    
    def queue_order(self, symbol: str, qty: int, side: str, 
                   order_type: str = "market", priority: int = 0) -> None:
        """
        Queue an order for batch execution.
        
        Args:
            symbol: Stock symbol
            qty: Quantity
            side: "buy" or "sell" 
            order_type: Order type
            priority: Order priority (lower = higher priority)
        """
        order = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'order_type': order_type,
            'priority': priority,
            'timestamp': time.time()
        }
        
        self.pending_orders.append(order)
        logger.debug(f"Queued order: {side.upper()} {qty} {symbol}")
    
    async def execute_batch(self, max_batch_size: int = 10) -> List[Dict]:
        """
        Execute queued orders in batches.
        
        Args:
            max_batch_size: Maximum orders per batch
            
        Returns:
            List of execution results
        """
        if not self.pending_orders:
            return []
        
        # Sort by priority and timestamp
        self.pending_orders.sort(key=lambda x: (x['priority'], x['timestamp']))
        
        results = []
        
        # Process in batches
        while self.pending_orders:
            batch = self.pending_orders[:max_batch_size]
            self.pending_orders = self.pending_orders[max_batch_size:]
            
            logger.info(f"Executing batch of {len(batch)} orders")
            
            # Submit batch
            batch_results = await self.broker.submit_multiple_orders_async(batch)
            results.extend(batch_results)
            
            # Rate limiting between batches
            if self.pending_orders and self.rate_limit_delay > 0:
                await asyncio.sleep(self.rate_limit_delay)
        
        successful_orders = sum(1 for result in results if result is not None)
        logger.info(f"Batch execution complete: {successful_orders}/{len(results)} orders succeeded")
        
        return results
    
    def clear_queue(self) -> int:
        """Clear pending orders queue."""
        count = len(self.pending_orders)
        self.pending_orders.clear()
        return count


# Usage example
async def parallel_trading_example():
    """Example of parallel trading operations."""
    
    # Initialize async components
    async_broker = AsyncAlpacaBroker(paper_trading=True, max_workers=6)
    batch_manager = BatchOrderManager(async_broker)
    
    try:
        # 1. Parallel data gathering
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        
        # Get account and positions concurrently
        account_task = async_broker.get_account_balance_async()
        positions_task = async_broker.get_positions_batch_async(symbols)
        
        account_info, positions = await asyncio.gather(account_task, positions_task)
        
        logger.info(f"Account equity: ${account_info.get('equity', 0):,.2f}")
        logger.info(f"Positions: {len([p for p in positions.values() if p])}")
        
        # 2. Queue multiple orders
        batch_manager.queue_order("AAPL", 10, "buy", priority=1)
        batch_manager.queue_order("MSFT", 15, "buy", priority=1) 
        batch_manager.queue_order("GOOGL", 5, "sell", priority=2)
        
        # 3. Execute orders in parallel
        results = await batch_manager.execute_batch(max_batch_size=5)
        
        logger.info(f"Executed {len(results)} orders")
        
    finally:
        async_broker.close()


if __name__ == "__main__":
    # Test the async broker
    asyncio.run(parallel_trading_example())