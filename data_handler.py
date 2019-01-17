from collections import deque
import pandas as pd
import numpy as np


def states_iterator(df):
    ticks = len(df)
    iterator = iter(df.values)
    return iterator, ticks


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


def close_price_iterator(df):
    close_price = pd.Series(df['Close'])
    iterator = iter(close_price)
    return iterator


def preprocess_data(data_set_file):
    data = pd.read_csv(data_set_file, index_col='dates')
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    data.drop_duplicates(inplace=True)
    return data
