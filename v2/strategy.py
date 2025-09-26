from v2.model import dummy_predict
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
def get_signal(historical_data) -> Signal: # some iterable, not sure what the model output will look like yet
    '''
    translate historical data into [BUY, SELL, HOLD] signal
    '''
    prediction = dummy_predict()        # log returns of close
    prediction = np.cumsum(prediction)  # maybe weight future predictions less

    if np.max(prediction) >= STRATEGY['buy_threshold_log']:
        return Signal.BUY
    elif np.min(prediction) <= STRATEGY['sell_threshold_log']:
        return Signal.SELL
    else:
        return Signal.HOLD
    

if __name__ == '__main__':
    prediction = dummy_predict()
    signal = get_signal(prediction)
    logger.debug(f'signal: {signal}')