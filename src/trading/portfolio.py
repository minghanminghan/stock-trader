from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.utils.logging_config import logger


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    symbol: str
    qty: int
    side: str  # 'long' or 'short'
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    entry_time: datetime


class Portfolio:
    """
    Track positions, PnL, and portfolio state.
    Maintains local state and syncs with broker when needed.
    """
    
    def __init__(self):
        """Initialize empty portfolio."""
        self.positions: Dict[str, Position] = {}
        self.realized_pl_today: float = 0.0
        self.total_trades_today: int = 0
        self.max_positions: int = 10
        
        logger.info("Portfolio initialized")
    
    def update_position_from_broker(self, symbol: str, broker_position: Dict) -> None:
        """
        Update position from broker data.
        
        Args:
            symbol: Stock symbol
            broker_position: Position dict from broker.get_position()
        """
        if broker_position is None:
            # No position exists, remove from local tracking
            if symbol in self.positions:
                del self.positions[symbol]
                logger.info(f"Position {symbol} closed")
            return
        
        # Create or update position
        self.positions[symbol] = Position(
            symbol=broker_position['symbol'],
            qty=broker_position['qty'],
            side=broker_position['side'],
            avg_entry_price=broker_position['avg_entry_price'],
            current_price=broker_position['market_value'] / broker_position['qty'] if broker_position['qty'] != 0 else 0,
            market_value=broker_position['market_value'],
            unrealized_pl=broker_position['unrealized_pl'],
            unrealized_plpc=broker_position['unrealized_plpc'],
            entry_time=datetime.now()  # Simplified - broker doesn't provide this easily
        )
        
        logger.debug(f"Position updated: {symbol} {broker_position['side']} {broker_position['qty']}")
    
    def update_all_positions_from_broker(self, broker_positions: List[Dict]) -> None:
        """
        Sync all positions from broker.
        
        Args:
            broker_positions: List of position dicts from broker.get_all_positions()
        """
        # Clear existing positions
        self.positions.clear()
        
        # Add all current positions
        for pos in broker_positions:
            self.positions[pos['symbol']] = Position(
                symbol=pos['symbol'],
                qty=pos['qty'],
                side=pos['side'],
                avg_entry_price=pos['avg_entry_price'],
                current_price=pos['market_value'] / pos['qty'] if pos['qty'] != 0 else 0,
                market_value=pos['market_value'],
                unrealized_pl=pos['unrealized_pl'],
                unrealized_plpc=pos['unrealized_plpc'],
                entry_time=datetime.now()
            )
        
        logger.info(f"Portfolio synced: {len(self.positions)} positions")
    
    def record_trade_execution(self, symbol: str, qty: int, side: str, 
                             fill_price: float, fill_time: datetime = None) -> None: # this is ok
        """
        Record a trade execution and update position.
        
        Args:
            symbol: Stock symbol
            qty: Quantity traded
            side: 'buy' or 'sell'
            fill_price: Execution price
            fill_time: Execution time (defaults to now)
        """
        if fill_time is None:
            fill_time = datetime.now()
        
        existing_pos = self.positions.get(symbol)
        
        if side.lower() == 'buy':
            if existing_pos is None:
                # New long position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    side='long',
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    market_value=qty * fill_price,
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                    entry_time=fill_time
                )
                logger.info(f"New position opened: {symbol} LONG {qty} @ {fill_price}")
            else:
                if existing_pos.side == 'long':
                    # Add to long position
                    total_qty = existing_pos.qty + qty
                    new_avg_price = ((existing_pos.qty * existing_pos.avg_entry_price) + 
                                   (qty * fill_price)) / total_qty
                    
                    existing_pos.qty = total_qty
                    existing_pos.avg_entry_price = new_avg_price
                    existing_pos.market_value = total_qty * existing_pos.current_price
                    
                    logger.info(f"Added to long position: {symbol} +{qty} @ {fill_price}")
                else:
                    # Covering short position
                    if qty >= abs(existing_pos.qty):
                        # Full cover + potential flip to long
                        realized_pl = abs(existing_pos.qty) * (existing_pos.avg_entry_price - fill_price)
                        self.realized_pl_today += realized_pl
                        
                        remaining_qty = qty - abs(existing_pos.qty)
                        if remaining_qty > 0:
                            # Flip to long
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                qty=remaining_qty,
                                side='long',
                                avg_entry_price=fill_price,
                                current_price=fill_price,
                                market_value=remaining_qty * fill_price,
                                unrealized_pl=0.0,
                                unrealized_plpc=0.0,
                                entry_time=fill_time
                            )
                            logger.info(f"Covered short and flipped to long: {symbol}")
                        else:
                            # Fully closed
                            del self.positions[symbol]
                            logger.info(f"Short position fully covered: {symbol}")
                    else:
                        # Partial cover
                        existing_pos.qty = -(abs(existing_pos.qty) - qty)
                        logger.info(f"Partially covered short: {symbol} -{qty}")
        
        elif side.lower() == 'sell':
            if existing_pos is None:
                # New short position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=-qty,
                    side='short',
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    market_value=-qty * fill_price,
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                    entry_time=fill_time
                )
                logger.info(f"New short position: {symbol} SHORT {qty} @ {fill_price}")
            else:
                if existing_pos.side == 'long':
                    # Selling long position
                    if qty >= existing_pos.qty:
                        # Full sale + potential flip to short
                        realized_pl = existing_pos.qty * (fill_price - existing_pos.avg_entry_price)
                        self.realized_pl_today += realized_pl
                        
                        remaining_qty = qty - existing_pos.qty
                        if remaining_qty > 0:
                            # Flip to short
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                qty=-remaining_qty,
                                side='short',
                                avg_entry_price=fill_price,
                                current_price=fill_price,
                                market_value=-remaining_qty * fill_price,
                                unrealized_pl=0.0,
                                unrealized_plpc=0.0,
                                entry_time=fill_time
                            )
                            logger.info(f"Sold long and flipped to short: {symbol}")
                        else:
                            # Fully closed
                            del self.positions[symbol]
                            logger.info(f"Long position fully sold: {symbol}")
                    else:
                        # Partial sale
                        existing_pos.qty -= qty
                        logger.info(f"Partially sold long: {symbol} -{qty}")
                else:
                    # Add to short position
                    total_qty = abs(existing_pos.qty) + qty
                    new_avg_price = ((abs(existing_pos.qty) * existing_pos.avg_entry_price) + 
                                   (qty * fill_price)) / total_qty
                    
                    existing_pos.qty = -total_qty
                    existing_pos.avg_entry_price = new_avg_price
                    existing_pos.market_value = -total_qty * existing_pos.current_price
                    
                    logger.info(f"Added to short position: {symbol} -{qty} @ {fill_price}")
        
        self.total_trades_today += 1
    
    def update_market_prices(self, price_data: Dict[str, float]) -> None:
        """
        Update current market prices for positions.
        
        Args:
            price_data: Dict mapping symbol -> current price
        """
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position.current_price = price_data[symbol]
                
                if position.side == 'long':
                    position.market_value = position.qty * price_data[symbol]
                    position.unrealized_pl = position.qty * (price_data[symbol] - position.avg_entry_price)
                    position.unrealized_plpc = (price_data[symbol] / position.avg_entry_price) - 1.0
                else:  # short
                    position.market_value = position.qty * price_data[symbol]  # qty is negative for short
                    position.unrealized_pl = abs(position.qty) * (position.avg_entry_price - price_data[symbol])
                    position.unrealized_plpc = (position.avg_entry_price / price_data[symbol]) - 1.0
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol."""
        return symbol in self.positions
    
    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_total_unrealized_pl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pl for pos in self.positions.values())
    
    def get_exposure(self, symbol: str) -> float:
        """Get market value exposure for a symbol."""
        position = self.positions.get(symbol)
        return abs(position.market_value) if position else 0.0
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary statistics."""
        return {
            'total_positions': len(self.positions),
            'portfolio_value': self.get_portfolio_value(),
            'unrealized_pl': self.get_total_unrealized_pl(),
            'realized_pl_today': self.realized_pl_today,
            'total_trades_today': self.total_trades_today,
            'positions': {symbol: {
                'qty': pos.qty,
                'side': pos.side,
                'avg_entry': pos.avg_entry_price,
                'current_price': pos.current_price,
                'unrealized_pl': pos.unrealized_pl,
                'unrealized_plpc': pos.unrealized_plpc
            } for symbol, pos in self.positions.items()}
        }