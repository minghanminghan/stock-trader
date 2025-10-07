from v2.model import predict_multiple_steps, StockPriceLSTM
from v2.config import STRATEGY
from v2.utils import log_params, logger
from typing import Literal
from enum import Enum
import numpy as np


class Signal(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


# TODO: translate model output into (SYMBOL, ACTION, QTY) tuple

# @log_params
def get_signal(prediction: np.ndarray) -> Signal:
    '''
    translate model prediction into [BUY, SELL, HOLD] signal

    Args:
        prediction: log_close predictions from predict_multiple_steps (shape: batch_size, n_steps)
    '''
    prediction = prediction.flatten()
    prediction = np.cumsum(prediction)  # cumulative log returns (total expected return)
    # logger.debug(f'log close prediction: {prediction}')

    if np.max(prediction) >= STRATEGY['buy_threshold_log']:
        return Signal.BUY
    elif np.min(prediction) <= STRATEGY['sell_threshold_log']:
        return Signal.SELL
    else:
        return Signal.HOLD


if __name__ == '__main__':
    from pprint import pprint
    from datetime import datetime
    from v2.utils import get_data, preprocess_data
    from v2.config import LSTM_MODEL
    from v2.model import create_StockPriceLSTM

    data = get_data('AAPL', datetime(2021, 1, 1), datetime(2021, 12, 31))
    preprocessed = preprocess_data(data)
    stock_slice = preprocessed.iloc[-LSTM_MODEL['input_length']:]
    model = create_StockPriceLSTM()
    prediction = predict_multiple_steps(model, stock_slice.to_numpy(), 15)
    # pprint(prediction)
    # prediction = dummy_predict()
    signal = get_signal(prediction)
    logger.debug(f'signal: {signal}')