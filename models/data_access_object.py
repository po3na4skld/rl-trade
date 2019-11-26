import pandas as pd
import numpy as np

from collections import deque


class DataAccessObject:

    def __init__(self, data_file, index_col=None, names=(), sort_idx=True, drop_duplicate=True, seq_len=20):
        if index_col and index_col in names:
            raise ValueError("Index column can't be in DataFrame columns")

        self.df = pd.read_csv(data_file, index_col=index_col)
        if names:
            self.df = self.df[names]
        if index_col:
            try:
                self.df.index = pd.to_datetime(self.df.index)
            except TypeError:
                print('DataFrame index not a datetime!')

        if sort_idx:
            self.df.sort_index(inplace=True)
        if drop_duplicate:
            self.df.drop_duplicates(inplace=True)

        self.ticks = len(self.df)
        self.close_price_iterator = self.get_col_iterator('Close')
        self.sequential_data, self.sequential_ticks = self.get_sequential_data(seq_len)

    def sample(self, n=10):
        return self.df.sample(n)

    def head(self, n=5):
        return self.df.head(n)

    def tail(self, n=5):
        return self.df.tail(n)

    def get_col_iterator(self, col_name):
        return iter(pd.Series(self.df[col_name]))

    def get_iterator(self):
        return iter(self.df.values)

    def get_sequential_data(self, seq_len=20):
        sequential_data = []
        prev_stocks = deque(maxlen=seq_len)
        for v in self.df.values:
            prev_stocks.append([n for n in v])
            if len(prev_stocks) == seq_len:
                sequential_data.append(np.array(prev_stocks))

        return iter(np.asarray(sequential_data)), len(sequential_data)


if __name__ == '__main__':
    dao = DataAccessObject('stocks/test_data.csv', index_col='dates', names=['Open', 'High', 'Low', 'Close', 'Volume'])
    print(dao.head())
