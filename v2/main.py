from v2.trader import OrderTracker, DataQueue, update_account, exec_buy_order, exec_sell_order
from v2.model import dummy_predict
from v2.strategy import get_signal
from v2.utils import logger, get_data
from v2.config import ALPACA_STREAM, ALPACA_REST, SYMBOLS

from pprint import pprint
from datetime import datetime, timedelta
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit


trackers = { symbol: OrderTracker(symbol) for symbol in SYMBOLS }
data_queues = { symbol: DataQueue(symbol) for symbol in SYMBOLS }

async def callback_bars(i):
    forecast = dummy_predict()


def main():
    # pre-populate data
    start_date = (datetime.now() - timedelta(days=60)).date().strftime('%Y-%m-%d')
    end_date = datetime.now().date().strftime('%Y-%m-%d')
    for symbol in SYMBOLS:
        
        data = get_data(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe_unit=TimeFrameUnit.Day,
        )
        data_queues[symbol].populate(data)
        print(data_queues[symbol].q)

    # websocket
    # ALPACA_STREAM.subscribe_bars(callback_bars, 'AAPL')
    # ALPACA_STREAM.run()


if __name__ == '__main__':
    main()