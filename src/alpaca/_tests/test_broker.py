#!/usr/bin/env python3
"""
Unit tests for src/alpaca/broker.py - Alpaca Trading Broker Wrapper
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
from alpaca.common.exceptions import APIError
from alpaca.trading.models import Order

from src.alpaca.broker import (
    AlpacaBroker, OrderRequest, OrderResult,
    create_buy_order, create_sell_order
)


class TestOrderRequest:
    """Test OrderRequest dataclass."""
    
    def test_order_request_creation(self):
        """Test OrderRequest creation with valid parameters."""
        order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy"
        )
        
        assert order.symbol == "SPY"
        assert order.qty == 10
        assert order.side == "buy"
        assert order.order_type == "market"
        assert order.limit_price is None
        assert order.time_in_force == "day"
    
    def test_order_request_with_limit(self):
        """Test OrderRequest with limit order parameters."""
        order = OrderRequest(
            symbol="AAPL",
            qty=5,
            side="sell",
            order_type="limit",
            limit_price=150.0,
            time_in_force="gtc"
        )
        
        assert order.symbol == "AAPL"
        assert order.qty == 5
        assert order.side == "sell"
        assert order.order_type == "limit"
        assert order.limit_price == 150.0
        assert order.time_in_force == "gtc"


class TestOrderResult:
    """Test OrderResult dataclass."""
    
    def test_order_result_success(self):
        """Test successful OrderResult creation."""
        result = OrderResult(
            success=True,
            order_id="123456",
            symbol="SPY",
            qty=10,
            side="buy",
            status="filled"
        )
        
        assert result.success is True
        assert result.order_id == "123456"
        assert result.symbol == "SPY"
        assert result.qty == 10
        assert result.side == "buy"
        assert result.status == "filled"
        assert result.error is None
    
    def test_order_result_failure(self):
        """Test failed OrderResult creation."""
        result = OrderResult(
            success=False,
            error="Insufficient buying power"
        )
        
        assert result.success is False
        assert result.error == "Insufficient buying power"
        assert result.order_id is None


class TestAlpacaBroker:
    """Test AlpacaBroker class."""
    
    @pytest.fixture
    def mock_trading_client(self):
        """Mock TradingClient."""
        with patch('src.alpaca.broker.TradingClient') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def broker(self, mock_trading_client):
        """Create test broker instance."""
        return AlpacaBroker(paper=True)
    
    def test_broker_initialization(self, mock_trading_client):
        """Test broker initialization."""
        broker = AlpacaBroker(paper=True)
        
        assert broker.paper is True
        assert broker.cache_ttl == 30
        assert broker._account_cache is None
        assert broker._positions_cache is None
        mock_trading_client.assert_called_once()
    
    def test_broker_initialization_live(self, mock_trading_client):
        """Test broker initialization for live trading."""
        broker = AlpacaBroker(paper=False)
        
        assert broker.paper is False
        mock_trading_client.assert_called_once()
    
    def test_validate_order_valid(self, broker):
        """Test order validation with valid order."""
        order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy"
        )
        
        assert broker._validate_order(order) is True
    
    def test_validate_order_invalid_symbol(self, broker):
        """Test order validation with invalid symbol."""
        order = OrderRequest(
            symbol="",
            qty=10,
            side="buy"
        )
        
        assert broker._validate_order(order) is False
    
    def test_validate_order_invalid_quantity(self, broker):
        """Test order validation with invalid quantity."""
        order = OrderRequest(
            symbol="SPY",
            qty=0,
            side="buy"
        )
        
        assert broker._validate_order(order) is False
    
    def test_validate_order_invalid_side(self, broker):
        """Test order validation with invalid side."""
        order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="invalid"
        )
        
        assert broker._validate_order(order) is False
    
    def test_validate_order_limit_without_price(self, broker):
        """Test order validation for limit order without price."""
        order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy",
            order_type="limit"
        )
        
        assert broker._validate_order(order) is False
    
    def test_validate_order_limit_with_price(self, broker):
        """Test order validation for limit order with price."""
        order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy",
            order_type="limit",
            limit_price=450.0
        )
        
        assert broker._validate_order(order) is True
    
    def test_place_order_success(self, broker):
        """Test successful order placement."""
        # Mock successful order response
        mock_order = Mock()
        mock_order.id = "12345"
        mock_order.status.value = "submitted"
        broker.client.submit_order.return_value = mock_order
        
        order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy"
        )
        
        result = broker.place_order(order)
        
        assert result.success is True
        assert result.order_id == "12345"
        assert result.symbol == "SPY"
        assert result.qty == 10
        assert result.side == "buy"
        assert result.status == "submitted"
    
    def test_place_order_validation_failure(self, broker):
        """Test order placement with validation failure."""
        order = OrderRequest(
            symbol="",  # Invalid symbol
            qty=10,
            side="buy"
        )
        
        result = broker.place_order(order)
        
        assert result.success is False
        assert result.error == "Invalid order parameters"
    
    def test_place_order_api_error(self, broker):
        """Test order placement with API error."""
        broker.client.submit_order.side_effect = APIError("API Error")
        
        order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy"
        )
        
        result = broker.place_order(order)
        
        assert result.success is False
        assert "Alpaca API error" in result.error
    
    def test_place_order_general_exception(self, broker):
        """Test order placement with general exception."""
        broker.client.submit_order.side_effect = Exception("Network error")
        
        order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy"
        )
        
        result = broker.place_order(order)
        
        assert result.success is False
        assert "Order placement error" in result.error
    
    def test_place_orders_batch(self, broker):
        """Test batch order placement."""
        # Mock successful order responses
        mock_order = Mock()
        mock_order.id = "12345"
        mock_order.status.value = "submitted"
        broker.client.submit_order.return_value = mock_order
        
        orders = [
            OrderRequest(symbol="SPY", qty=10, side="buy"),
            OrderRequest(symbol="AAPL", qty=5, side="sell")
        ]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            results = broker.place_orders(orders)
        
        assert len(results) == 2
        assert all(result.success for result in results)
    
    def test_get_account_success(self, broker):
        """Test successful account information retrieval."""
        # Mock account response
        mock_account = Mock()
        mock_account.equity = 10000.0
        mock_account.cash = 5000.0
        mock_account.buying_power = 20000.0
        mock_account.portfolio_value = 10000.0
        mock_account.pattern_day_trader = False
        mock_account.trading_blocked = False
        mock_account.account_blocked = False
        
        broker.client.get_account.return_value = mock_account
        
        account = broker.get_account()
        
        assert account is not None
        assert account['equity'] == 10000.0
        assert account['cash'] == 5000.0
        assert account['buying_power'] == 20000.0
        assert account['portfolio_value'] == 10000.0
        assert account['pattern_day_trader'] is False
        assert 'last_updated' in account
    
    def test_get_account_cached(self, broker):
        """Test account information with caching."""
        # Set up cache
        broker._account_cache = {'equity': 10000.0}
        broker._cache_timestamp = datetime.now()
        
        account = broker.get_account(use_cache=True)
        
        assert account == {'equity': 10000.0}
        # Should not call API when using cache
        broker.client.get_account.assert_not_called()
    
    def test_get_account_error(self, broker):
        """Test account information retrieval error."""
        broker.client.get_account.side_effect = Exception("API Error")
        
        account = broker.get_account()
        
        assert account is None
    
    def test_get_positions_success(self, broker):
        """Test successful positions retrieval."""
        # Mock positions response
        mock_position = Mock()
        mock_position.symbol = "SPY"
        mock_position.qty = 10
        mock_position.market_value = 4500.0
        mock_position.cost_basis = 4400.0
        mock_position.unrealized_pl = 100.0
        mock_position.unrealized_plpc = 0.0227
        mock_position.avg_entry_price = 440.0
        
        broker.client.get_all_positions.return_value = [mock_position]
        
        positions = broker.get_positions()
        
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) == 1
        assert positions.iloc[0]['symbol'] == "SPY"
        assert positions.iloc[0]['qty'] == 10
        assert positions.iloc[0]['side'] == "long"
    
    def test_get_positions_empty(self, broker):
        """Test positions retrieval with no positions."""
        broker.client.get_all_positions.return_value = []
        
        positions = broker.get_positions()
        
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) == 0
        expected_columns = ['symbol', 'qty', 'side', 'market_value', 
                          'cost_basis', 'unrealized_pl', 'unrealized_plpc']
        assert list(positions.columns) == expected_columns
    
    def test_get_positions_short(self, broker):
        """Test positions retrieval with short position."""
        # Mock short position
        mock_position = Mock()
        mock_position.symbol = "SPY"
        mock_position.qty = -10  # Short position
        mock_position.market_value = -4500.0
        mock_position.cost_basis = 4400.0
        mock_position.unrealized_pl = -100.0
        mock_position.unrealized_plpc = -0.0227
        mock_position.avg_entry_price = 440.0
        
        broker.client.get_all_positions.return_value = [mock_position]
        
        positions = broker.get_positions()
        
        assert positions.iloc[0]['side'] == "short"
        assert positions.iloc[0]['qty'] == -10
    
    def test_get_positions_error(self, broker):
        """Test positions retrieval error."""
        broker.client.get_all_positions.side_effect = Exception("API Error")
        
        positions = broker.get_positions()
        
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) == 0
    
    def test_get_position_exists(self, broker):
        """Test getting specific position that exists."""
        # Mock positions DataFrame
        mock_positions = pd.DataFrame([{
            'symbol': 'SPY',
            'qty': 10,
            'side': 'long',
            'market_value': 4500.0
        }])
        
        with patch.object(broker, 'get_positions', return_value=mock_positions):
            position = broker.get_position('SPY')
        
        assert position is not None
        assert position['symbol'] == 'SPY'
        assert position['qty'] == 10
    
    def test_get_position_not_exists(self, broker):
        """Test getting specific position that doesn't exist."""
        mock_positions = pd.DataFrame([{
            'symbol': 'AAPL',
            'qty': 5,
            'side': 'long',
            'market_value': 750.0
        }])
        
        with patch.object(broker, 'get_positions', return_value=mock_positions):
            position = broker.get_position('SPY')
        
        assert position is None
    
    def test_get_position_empty_positions(self, broker):
        """Test getting position when no positions exist."""
        mock_positions = pd.DataFrame()
        
        with patch.object(broker, 'get_positions', return_value=mock_positions):
            position = broker.get_position('SPY')
        
        assert position is None
    
    def test_get_buying_power(self, broker):
        """Test getting buying power."""
        mock_account = {'buying_power': 20000.0}
        
        with patch.object(broker, 'get_account', return_value=mock_account):
            buying_power = broker.get_buying_power()
        
        assert buying_power == 20000.0
    
    def test_get_buying_power_no_account(self, broker):
        """Test getting buying power when account unavailable."""
        with patch.object(broker, 'get_account', return_value=None):
            buying_power = broker.get_buying_power()
        
        assert buying_power is None
    
    def test_get_portfolio_value(self, broker):
        """Test getting portfolio value."""
        mock_account = {'portfolio_value': 10000.0}
        
        with patch.object(broker, 'get_account', return_value=mock_account):
            portfolio_value = broker.get_portfolio_value()
        
        assert portfolio_value == 10000.0
    
    def test_can_place_order_buy_sufficient_power(self, broker):
        """Test can place buy order with sufficient buying power."""
        order = OrderRequest(symbol="SPY", qty=1, side="buy")
        
        with patch.object(broker, 'get_buying_power', return_value=1000.0):
            can_place = broker.can_place_order(order)
        
        assert can_place is True
    
    def test_can_place_order_buy_insufficient_power(self, broker):
        """Test can place buy order with insufficient buying power."""
        order = OrderRequest(symbol="SPY", qty=100, side="buy")
        
        with patch.object(broker, 'get_buying_power', return_value=500.0):
            can_place = broker.can_place_order(order)
        
        assert can_place is False
    
    def test_can_place_order_sell_with_position(self, broker):
        """Test can place sell order with sufficient position."""
        order = OrderRequest(symbol="SPY", qty=5, side="sell")
        mock_position = {'qty': 10}
        
        with patch.object(broker, 'get_position', return_value=mock_position):
            can_place = broker.can_place_order(order)
        
        assert can_place is True
    
    def test_can_place_order_sell_insufficient_position(self, broker):
        """Test can place sell order with insufficient position."""
        order = OrderRequest(symbol="SPY", qty=15, side="sell")
        mock_position = {'qty': 10}
        
        with patch.object(broker, 'get_position', return_value=mock_position):
            can_place = broker.can_place_order(order)
        
        assert can_place is False
    
    def test_can_place_order_sell_no_position(self, broker):
        """Test can place sell order with no position."""
        order = OrderRequest(symbol="SPY", qty=5, side="sell")
        
        with patch.object(broker, 'get_position', return_value=None):
            can_place = broker.can_place_order(order)
        
        assert can_place is False
    
    def test_can_place_order_no_buying_power_info(self, broker):
        """Test can place order when buying power unavailable."""
        order = OrderRequest(symbol="SPY", qty=1, side="buy")
        
        with patch.object(broker, 'get_buying_power', return_value=None):
            can_place = broker.can_place_order(order)
        
        assert can_place is False
    
    def test_close_position_long(self, broker):
        """Test closing long position."""
        mock_position = {'qty': 10, 'symbol': 'SPY'}
        mock_order = Mock()
        mock_order.id = "12345"
        mock_order.status.value = "submitted"
        
        broker.client.submit_order.return_value = mock_order
        
        with patch.object(broker, 'get_position', return_value=mock_position):
            result = broker.close_position('SPY')
        
        assert result.success is True
        assert result.symbol == 'SPY'
        # Should place sell order to close long position
        broker.client.submit_order.assert_called_once()
    
    def test_close_position_short(self, broker):
        """Test closing short position."""
        mock_position = {'qty': -10, 'symbol': 'SPY'}
        mock_order = Mock()
        mock_order.id = "12345"
        mock_order.status.value = "submitted"
        
        broker.client.submit_order.return_value = mock_order
        
        with patch.object(broker, 'get_position', return_value=mock_position):
            result = broker.close_position('SPY')
        
        assert result.success is True
        # Should place buy order to close short position
        broker.client.submit_order.assert_called_once()
    
    def test_close_position_not_exists(self, broker):
        """Test closing position that doesn't exist."""
        with patch.object(broker, 'get_position', return_value=None):
            result = broker.close_position('SPY')
        
        assert result.success is False
        assert "No position found" in result.error
    
    def test_cache_validity(self, broker):
        """Test cache validity checking."""
        # No cache timestamp
        assert broker._is_cache_valid() is False
        
        # Recent timestamp
        broker._cache_timestamp = datetime.now()
        assert broker._is_cache_valid() is True
        
        # Old timestamp
        from datetime import timedelta
        broker._cache_timestamp = datetime.now() - timedelta(seconds=100)
        assert broker._is_cache_valid() is False


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_buy_order(self):
        """Test create_buy_order convenience function."""
        order = create_buy_order("SPY", 10)
        
        assert order.symbol == "SPY"
        assert order.qty == 10
        assert order.side == "buy"
        assert order.order_type == "market"
    
    def test_create_sell_order(self):
        """Test create_sell_order convenience function."""
        order = create_sell_order("AAPL", 5)
        
        assert order.symbol == "AAPL"
        assert order.qty == 5
        assert order.side == "sell"
        assert order.order_type == "market"


if __name__ == "__main__":
    pytest.main([__file__])