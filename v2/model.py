from v2.config import LSTM_MODEL, RANDOM_SEED
from v2.utils import log_params, logger
import pandas as pd
import numpy as np
import random
random.seed(RANDOM_SEED)


output_length = LSTM_MODEL['output_length']

@log_params
def dummy_predict(*args, **kwargs):
    '''
    returns mock log returns (log c_{t+1} - log c_t) of length 1 * LSTM_MODEL.output_length (predicting close for next n days)
    '''
    prices = [100 + 5 * round(random.random(), 2) for _ in range(output_length)]
    return np.diff(np.log(prices))


if __name__ == '__main__':
    from pprint import pprint
    from alpaca_trade_api.rest import TimeFrameUnit
    from v2.utils import get_data

    data = get_data('AAPL', "2021-06-01", "2021-06-30", TimeFrameUnit.Day)

    prediction = dummy_predict()
    pprint(prediction)

