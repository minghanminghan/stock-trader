from v2.config import ALPACA_CLIENT, ALPACA_STREAM, STRATEGY, ENVIRONMENT, LSTM_MODEL
from v2.utils import log_params, log_params_async, logger, preprocess_data
from pprint import pprint
from typing import Literal, Optional
# from alpaca_trade_api.rest import Order
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.models import Quote
from dataclasses import dataclass
from pandas import DataFrame
import numpy as np
import heapq


@dataclass
class SimpleOrder:
    '''
    order_id: order uuid\n
    price: price of stock at buy order\n
    shares: number of shares bought\n
    tp: take profit (derived)\n
    sl: stop loss (derived)
    '''
    order_id: str
    price: float    # price at time bought
    shares: float   # nominal shares bought
    tp: float       # read from config
    sl: float       # ^
    
    def __init__(self, order_id: str, price: float, shares: float):
        self.order_id = order_id
        self.price = price
        self.shares = shares
        self.tp = shares * STRATEGY['take_profit']
        self.sl = shares * STRATEGY['stop_loss']

    def __lt__(self, other):
        '''
        used for heap push/pop
        '''
        return self.price > other.price
    

class OrderTracker:
    '''
    class to track orders and compare to bid/ask price
    '''
    def __init__(self, symbol: str, bid_price: float = 0.0, ask_price: float = 0.0):
        self.bid_price: float = bid_price # = quote.bid_price # will constantly update this from websocket
        self.ask_price: float = ask_price # = quote.ask_price
        self.symbol: str = symbol
        self.orders: list[SimpleOrder] = []

    def update_quote(self, quote: Quote):
        self.bid_price = quote.bid_price
        self.ask_price = quote.ask_price

    def update_quote_backtest(self, price: float): # for backtesting
        self.bid_price = price
        self.ask_price = price

    def add_order(self, order_id: str, shares: float):
        order = SimpleOrder(
            order_id,
            self.bid_price,
            shares,
        )
        heapq.heappush(self.orders, order)

    # def check_exits(self) -> list[SimpleOrder]:
    #     '''
    #     price: new stock price\n
    #     returns the tp and sl orders that cross this threshold
    #     '''
    #     valid_orders: list[SimpleOrder] = []
    #     for i in self.orders: # can improve this
    #         if i.tp <= self.bid_price or i.sl >= self.ask_price:
    #             valid_orders.append(i)
    #     return valid_orders
    
    def execute_exits(self) -> list[SimpleOrder]:
        '''
        price: new stock price\n
        remove orders past threshold and return the notional value for the sell order
        '''
        exits = [order for order in self.orders if self.ask_price >= order.tp or self.ask_price <= order.sl]
        remaining = [order for order in self.orders if not(self.ask_price >= order.tp or self.ask_price <= order.sl)]

        self.orders = remaining
        return exits

    def get_total_shares(self) -> float:
        """Get total shares across all open positions."""
        return sum(order.shares for order in self.orders)

    def has_positions(self) -> bool:
        """Check if there are any open positions."""
        return len(self.orders) > 0

    def save_state(self): # save to pickle
        pass

    def load_state(self):
        pass


class DataQueue():
    '''
    class to track historical data for model forecasting
    '''
    def __init__(self, symbol: str, max_size: int = LSTM_MODEL['input_length']):
        self.symbol = symbol
        self.max_size = max_size
        self.q: DataFrame
    
    def populate(self, bars: DataFrame):
        self.q = preprocess_data(bars) # maybe slice to only keep the end
    
    def update(self, bar: dict): # 1 time slice
        if len(self.q) >= self.max_size: # pop
            self.q.drop(self.q.index[0], inplace=True)
        self.q.loc[len(self.q)] = bar


@log_params
def update_account():
    account = ALPACA_CLIENT.get_account()

    # set STRATEGY params (buying power, cash)
    logger.debug(f'buying power: {account.buying_power}, cash: {account.cash}')
    STRATEGY['buying_power'] = account.buying_power
    STRATEGY['cash'] = account.cash


@log_params
def exec_order(symbol: str, side: Literal['buy', 'sell'], notional: float):
    '''
    symbol: symbol to buy\n
    entry_price: market price at buy time, used to calculate tp, sl\n
    notional: notional shares
    '''
    return ALPACA_CLIENT.submit_order(MarketOrderRequest(
        symbol=symbol,
        side=side,
        notional=notional,
    )
)


# @log_params_async
async def dummy_callback(foo):
    pprint(foo)


if __name__ == '__main__':
    update_account()
    # aapl_latest = ALPACA_REST.get_latest_trade('AAPL')
    # print(aapl_latest)
    # order = exec_buy_order('AAPL', aapl_latest.p, 100)
    # print(order)

    # orders = ALPACA_REST.list_orders(status='all', symbols=['AAPL'])
    # pprint(orders)

    # ALPACA_REST.submit_order('AAPL', )

    # update_account()
    # ALPACA_STREAM.subscribe_bars(dummy_callback, 'AAPL')
    # ALPACA_STREAM.subscribe_trades(dummy_callback, 'AAPL')
    # ALPACA_STREAM.run()

    pass