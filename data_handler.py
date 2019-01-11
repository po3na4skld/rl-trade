from collections import deque
import pandas as pd
import numpy as np


def create_sequences(df):
    seq_len = 20
    sequential_data = []
    prev_stocks = deque(maxlen=seq_len)
    for v in df.values:
        prev_stocks.append([n for n in v])
        if len(prev_stocks) == seq_len:
            sequential_data.append(np.array(prev_stocks))
    ticks = np.asarray(sequential_data).shape[0]
    iterator = iter(np.asarray(sequential_data))
    return iterator, ticks


def create_price_frame(df):
    price_close = pd.Series(df['Close'])
    iterator = iter(price_close)
    return iterator


def preprocess_data(data_set_file):
    state_frame = pd.read_csv(data_set_file)

    state_frame.index = pd.to_datetime(state_frame['Date'])
    state_frame.index.name = 'dates'
    state_frame = state_frame.drop('Date', axis=1)

    state_frame.dropna(inplace=True)

    state_frame['AO'] = state_frame['AO'].pct_change()

    state_frame.dropna(inplace=True)
    return state_frame
