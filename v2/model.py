from datetime import datetime
from v2.config import LSTM_MODEL, LSTM_TRAINING, RANDOM_SEED
from v2.utils import log_params, logger
import os
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


# @log_params
def predict_multiple_steps(model: StockPriceLSTM, initial_sequence: np.ndarray, n_steps: int=LSTM_MODEL['output_length']):
    tensor = torch.from_numpy(initial_sequence).float()

    # Add batch dimension if needed (2D -> 3D)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # (seq_len, features) -> (1, seq_len, features)

    # Ensure it's on the correct device
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    batch_size = tensor.size(0)
    # seq_len = tensor.size(1)
    # input_size = tensor.size(2)

    # Pre-allocate prediction tensor
    log_close_predictions = torch.zeros(batch_size, n_steps, device=tensor.device)

    # Pre-allocate full sequence tensor to avoid repeated concatenation
    # This holds the sliding window that gets updated each step
    current_input = tensor.clone()  # (batch_size, seq_len, input_size)

    for step in range(n_steps):
        # Predict next timestep (all features)
        next_features = model(current_input)  # (batch_size, input_size)

        # Store log_close prediction directly in pre-allocated tensor
        log_close_predictions[:, step] = next_features[:, 0]

        # Update input sequence in-place using slicing (more efficient than cat)
        if step < n_steps - 1:  # Skip on last iteration to avoid unnecessary work
            # Shift the sequence: move elements left by one position
            current_input[:, :-1, :] = current_input[:, 1:, :]
            # Add new prediction at the end
            current_input[:, -1, :] = next_features

    return log_close_predictions.detach().numpy()  # (batch_size, n_steps)


@log_params
def create_StockPriceLSTM(path: str|None=None) -> StockPriceLSTM:
    '''
    path: relative path from cwd to state dict
    '''
    model = StockPriceLSTM(
        input_size=LSTM_MODEL['input_size'],
        output_size=LSTM_MODEL['output_size'],
        hidden_size=LSTM_MODEL['hidden_size'],
        num_layers=LSTM_MODEL['num_layers'],
        dropout=LSTM_MODEL['dropout'],
        input_length=LSTM_MODEL['input_length'],
        output_length=LSTM_MODEL['output_length']
    )
    try:
        if path != None:
            model.load_state_dict(torch.load(os.path.join(os.getcwd(), path)))
        else:
            DEFAULT_PATH = os.path.join(os.getcwd(), 'weights', 'best_model.pth')
            model.load_state_dict(torch.load(DEFAULT_PATH))
        model.eval()
        return model
    except:
        raise FileNotFoundError(f"Error loading state dict from path: {path}")


@log_params
def dummy_predict(*args, **kwargs):
    '''
    returns mock log returns (log c_{t+1} - log c_t) of length 1 * LSTM_MODEL.output_length (predicting close for next n days)
    '''
    prices = [100 + 5 * round(random.random(), 2) for _ in range(LSTM_MODEL['output_length'])]
    return np.diff(np.log(prices))


if __name__ == '__main__':
    from pprint import pprint
    from v2.utils import get_data, preprocess_data
    from v2.config import LSTM_MODEL

    data = get_data('AAPL', datetime(2021, 1, 1), datetime(2021, 12, 31))
    preprocessed = preprocess_data(data)
    stock_slice = preprocessed.iloc[-LSTM_MODEL['input_length']:]
    model = create_StockPriceLSTM()
    prediction = predict_multiple_steps(model, stock_slice.to_numpy(), 15)
    pprint(prediction)

    # prediction = dummy_predict()
    # pprint(prediction)