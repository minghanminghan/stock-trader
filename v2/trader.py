from v2.config import ALPACA_REST, ALPACA_STREAM, STRATEGY, ENVIRONMENT, LSTM_MODEL
from v2.utils import log_params, log_params_async, logger, preprocess_data
from pprint import pprint
from typing import Optional
from alpaca_trade_api.rest import Order
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
        self.symbol: str = symbol
        self.orders: list[SimpleOrder] = []

    def add_order(self, order_id: str, price: float, shares: float):
        order = SimpleOrder(
            order_id,
            price,
            shares,
            tp=price * STRATEGY['take_profit'],
            sl=price * STRATEGY['stop_loss']
        )
        bisect.insort(self.orders, order)

    def check_orders(self, price: float) -> list[SimpleOrder]:
        '''
        price: new stock price\n
        returns the tp and sl orders that cross this threshold
        '''
        orders: list[SimpleOrder] = []
        for i in self.orders: # can improve this
            if i.tp <= price or i.sl >= price:
                orders.append(i)
        return orders
    
    def get_sell_order_value(self, price: float) -> float:
        '''
        price: new stock price\n
        remove orders past threshold and return the notional value for the sell order
        '''
        shares = 0
        remaining_orders = []

        for order in self.orders:
            if price >= order.tp or price <= order.sl:
                shares += order.shares
            else:
                remaining_orders.append(order)

        self.orders = remaining_orders
        return shares * price

    def save_state(self):
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
    account = ALPACA_REST.get_account()

    # set STRATEGY params (buying power, cash)
    logger.debug(f'buying power: {account.buying_power}, cash: {account.cash}')
    STRATEGY['buying_power'] = account.buying_power
    STRATEGY['cash'] = account.cash


@log_params
def exec_buy_order(symbol: str, price: float, notional: float = STRATEGY['cash'] * 0.02):
    '''
    symbol: symbol to buy\n
    entry_price: market price at buy time, used to calculate tp, sl\n
    notional: notional shares
    '''
    # add to resp. OrderTracker
    return ALPACA_REST.submit_order(
        symbol=symbol,
        side='buy',
        notional=notional,
    )


@log_params
def exec_sell_order(symbol: str, notional: float):
    '''
    symbol: symbol to sell\n
    notional: amount to sell\n
    '''
    return ALPACA_REST.submit_order(
        symbol=symbol,
        side='sell',
        notional=notional,
    )


def get_order(order_id: str, nested: bool = False) -> Order:
    return ALPACA_REST.get_order(order_id=order_id, nested=nested)


# @log_params_async
async def dummy_callback(foo):
    pprint(foo)


if __name__ == '__main__':
    update_account()
    # aapl_latest = ALPACA_REST.get_latest_trade('AAPL')
    # print(aapl_latest)
    # order = exec_buy_order('AAPL', aapl_latest.p, 100)
    # print(order)

    # order = get_order('54868f24-d5fb-467b-acc9-89f1507f329a', True)
    # pprint(order)
    # pprint(order.stop_price)

    # orders = ALPACA_REST.list_orders(status='all', symbols=['AAPL'])
    # pprint(orders)

    # ALPACA_REST.submit_order('AAPL', )

    # update_account()
    # ALPACA_STREAM.subscribe_bars(dummy_callback, 'AAPL')
    # ALPACA_STREAM.subscribe_trades(dummy_callback, 'AAPL')
    # ALPACA_STREAM.run()

    pass