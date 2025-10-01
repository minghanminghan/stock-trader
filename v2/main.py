from v2.trader import OrderTracker, DataQueue, update_account, exec_order
from v2.model import create_StockPriceLSTM, predict_multiple_steps
from v2.strategy import get_signal, Signal
from v2.utils import logger, get_data
from v2.config import ALPACA_STREAM, ALPACA_CLIENT, STRATEGY, SYMBOLS

from pprint import pprint
from datetime import datetime, timedelta
from alpaca.data.models import Bar, Quote
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


order_trackers = { symbol: OrderTracker(symbol) for symbol in SYMBOLS }
data_queues = { symbol: DataQueue(symbol) for symbol in SYMBOLS }
model = create_StockPriceLSTM()

async def callback_bar(bar: Bar): # may be incorrect
    symbol = bar.symbol
    cur = data_queues[symbol]
    cur.update(bar.model_dump())

    forecast = predict_multiple_steps(model, cur.q.to_numpy())
    signal = get_signal(forecast)
    
    if signal == Signal.BUY:
        order = exec_order(symbol, 'buy', STRATEGY['cash'] * 0.02)
        logger.debug(order)
        # TODO: append to some list
        
    elif signal == Signal.SELL: # get 
        sell_amount = order_trackers[symbol].get_sell_order_value()
        order = exec_order(symbol, 'sell', sell_amount)
        logger.debug(order)
        # TODO: append to some list

    else:
        logger.debug('no signal generated')
        pass


async def callback_quote(quote: Quote):
    symbol = quote.symbol
    order_trackers[symbol].update_quote(quote)


def main():
    # pre-populate data
    start_date = (datetime.now() - timedelta(days=60))
    end_date = datetime.now()
    for symbol in SYMBOLS:
        data = get_data(
            symbol=symbol,
            start=start_date,
            end=end_date,
        )
        data_queues[symbol].populate(data)
        print(data_queues[symbol].q)

    # websocket
    ALPACA_STREAM.subscribe_bars(callback_bar, *SYMBOLS)
    # ALPACA_STREAM.subscribe_quotes(, *SYMBOLS)
    ALPACA_STREAM.run()


if __name__ == '__main__':
    main()