from datetime import datetime
from v2.config import LSTM_MODEL, LSTM_TRAINING, RANDOM_SEED
from v2.utils import log_params, logger
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import random
random.seed(RANDOM_SEED)


class StockPriceLSTM(nn.Module):
    def __init__(self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: int,
        input_length: int,
        output_length: int,
    ):
        super(StockPriceLSTM, self).__init__()
        self.input_size = input_size
        self.input_length = input_length
        self.output_size = output_size
        self.output_length = output_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass input through the LSTM layer
        # output: (batch_size, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Use only the final timestep for prediction
        final_hidden = out[:, -1, :]  # (batch_size, hidden_size)

        # Single step prediction of all input features
        prediction = self.fc(final_hidden)  # (batch_size, input_size)
        return prediction


def predict_multiple_steps(model, initial_sequence, n_steps):
    predictions = []
    current_input = initial_sequence  # (batch_size, seq_len, input_size)

    for _ in range(n_steps):
        # Predict next timestep (all features)
        next_features = model(current_input)  # (batch_size, input_size)
        predictions.append(next_features)

        # Update input: remove oldest timestep, add new prediction
        current_input = torch.cat([
            current_input[:, 1:, :],  # Remove first timestep
            next_features.unsqueeze(1)  # Add predicted timestep
        ], dim=1)

    return torch.stack(predictions, dim=1)  # (batch_size, n_steps, input_size)

# TODO:
# - implement training (sampling, loss, validation, optimizing training)
# - write unit tests
# - backtest (involves more extensive mocking)

@log_params
def dummy_predict(*args, **kwargs):
    '''
    returns mock log returns (log c_{t+1} - log c_t) of length 1 * LSTM_MODEL.output_length (predicting close for next n days)
    '''
    prices = [100 + 5 * round(random.random(), 2) for _ in range(LSTM_MODEL['output_length'])]
    return np.diff(np.log(prices))


if __name__ == '__main__':
    from pprint import pprint
    from alpaca.data.timeframe import TimeFrameUnit
    from v2.utils import get_data

    data = get_data('AAPL', datetime(2021, 1, 1), datetime(2021, 12, 31))
    prediction = dummy_predict()
    pprint(prediction)

