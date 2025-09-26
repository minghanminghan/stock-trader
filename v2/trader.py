from v2.config import ALPACA_CLIENT, ALPACA_STREAM, STRATEGY, ENVIRONMENT, LSTM_MODEL
from v2.utils import log_params, log_params_async, logger, preprocess_data
from pprint import pprint
from typing import Literal
# from alpaca_trade_api.rest import Order
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.models import Quote
from dataclasses import dataclass
from pandas import DataFrame
import bisect


@dataclass
class SimpleOrder: # potentially deconstruct into numpy array
    '''
    order_id: order uuid\n
    price: price of stock at buy order\n
    shares: number of shares bought\n
    tp: take profit (derived)\n
    sl: stop loss (derived)
    '''
    order_id: str
    price: float
    shares: float
    tp: float
    sl: float
    
    def __lt__(self, other):
        return self.price < other.price
    

class OrderTracker:
    def __init__(self, symbol: str):
        self.bid_price: float # = quote.bid_price # will constantly update this from websocket
        self.ask_price: float # = quote.ask_price
        self.symbol: str = symbol
        self.orders: list[SimpleOrder] = []

    def update_quote(self, quote: Quote):
        self.bid_price = quote.bid_price
        self.ask_price = quote.ask_price

    def add_order(self, order_id: str, shares: float):
        order = SimpleOrder(
            order_id,
            self.bid_price,
            shares,
            tp=self.bid_price * STRATEGY['take_profit'],
            sl=self.bid_price * STRATEGY['stop_loss']
        )
        bisect.insort(self.orders, order)

    def check_orders(self) -> list[SimpleOrder]:
        '''
        price: new stock price\n
        returns the tp and sl orders that cross this threshold
        '''
        orders: list[SimpleOrder] = []
        for i in self.orders: # can improve this
            if i.tp <= self.bid_price or i.sl >= self.ask_price:
                orders.append(i)
        return orders
    
    def get_sell_order_value(self) -> float:
        '''
        price: new stock price\n
        remove orders past threshold and return the notional value for the sell order
        '''
        shares = 0
        remaining_orders = []

        for order in self.orders:
            if self.ask_price >= order.tp or self.ask_price <= order.sl:
                shares += order.shares
            else:
                remaining_orders.append(order)

        self.orders = remaining_orders
        return shares * self.ask_price

    def save_state(self): # save to pickle
        pass

    def load_state(self):
        pass



class DataQueue():
    def __init__(self, symbol: str, max_size: int = LSTM_MODEL['input_length']):
        self.symbol = symbol
        self.max_size = max_size
        self.q: DataFrame
    
    def populate(self, bars: DataFrame):
        self.q = preprocess_data(bars)
    
    def update(self, bar: dict): # might be Dataframe
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