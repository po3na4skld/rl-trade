from models.data_access_object import DataAccessObject
from models.portfolio import Portfolio

import numpy as np
import gym


class Environment(gym.Env):

    def __init__(self, started_money, data, risk_level=0.1, index_col='', names=(), seq_len=20):
        self.data = data
        self.index_col = index_col
        self.names = names
        self.seq_len = seq_len

        if risk_level > 1.0 or not isinstance(risk_level, float):
            raise ValueError('Risk can be only in 0.01..1.0 range.')
        self.risk_level = risk_level

        self.portfolio = Portfolio(started_money)

        self.dao = None
        self.done = None

        self.current_state = None
        self.current_price = None

        self.reset()

        self.action_space = gym.spaces.Discrete(3)
        self.actions = {0: self.buy, 1: self.sell, 2: self.hold}
        # TODO: define observation space
        # self.observation_space = gym.spaces.Discrete()

    def buy(self, count):
        if self.portfolio.bankroll < count * self.current_price:
            return -1

        return self.portfolio.open_order('ETH', count, self.current_price)

    def sell(self):
        if not self.portfolio.opened_orders:
            return -1

        return self.portfolio.close_order(self.current_price)

    def hold(self):
        if self.portfolio.opened_orders:
            buy_price = self.portfolio.opened_orders[0]['buy_price']
            price_diff = np.abs((buy_price - self.current_price) / ((buy_price + self.current_price) / 2))
            if price_diff > 0.2:  # add this to a Strategy model
                return -1

        return 0

    def reset(self):
        self.dao = DataAccessObject(self.data, self.index_col, self.names, self.seq_len)
        self.done = False

    def update_state(self):
        try:
            self.current_state = next(self.dao.sequential_data)
            self.current_price = next(self.dao.close_price_iterator)
        except StopIteration:
            self.done = True

    @staticmethod
    def _clip_reward(reward):
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        else:
            reward = 0

        return reward

    def step(self, action):
        if action == 0:  # buy
            buy_count = (self.portfolio.bankroll * self.risk_level) / self.current_price
            reward = self.buy(buy_count)
            self.update_state()
        elif action == 1:  # sell
            reward = self.sell()
            self.update_state()
        else:  # hold
            reward = self.hold()
            self.update_state()

        return self.current_state, self._clip_reward(reward), self.done, self.portfolio.__str__()
