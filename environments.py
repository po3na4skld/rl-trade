from agents import QLearningLSTM, QLearningNN, Reinforce
from data_handler import *
import numpy as np


class Environment:

    def __init__(self, data_file, start_money):

        self.money_at_start = start_money
        self.money = start_money
        self.orders = {}

        self.df = preprocess_data(data_file)

        self.price_iter = close_price_iterator(self.df)
        self.current_price = next(self.price_iter)

    def buy(self, count, price):
        if self.money < count * price:
            pass
        else:
            self.money -= (count * price)
            if price in self.orders.keys():
                self.orders[price] += count
            else:
                self.orders.update([(price, count)])

    def sell(self, price):
        self.money += sum(self.orders.values()) * price
        self.orders.clear()

    def hold(self):
        pass

    def __repr__(self):
        return 'Total profit: {:.2f} '.format(self.money - self.money_at_start)


class QLearningEnvironment(Environment):

    def __init__(self, data_file, start_money, model_type, gamma, epsilon):
        super().__init__(data_file, start_money)

        if model_type == 'NN':
            self.state_iter, self.ticks = states_iterator(self.df)
        else:
            self.state_iter, self.ticks = create_sequences(self.df)

        self.current_state = next(self.state_iter)
        self.done = False
        self.state_dim = self.current_state.shape
        self.actions = {0: self.hold, 1: self.buy, 2: self.sell}
        self.action_space = len(self.actions)

        if model_type == 'NN':
            self.agent = QLearningNN(self.action_space, self.state_dim, gamma, epsilon)
            self.model = self.agent.model
        else:
            self.agent = QLearningLSTM(self.action_space, self.state_dim, gamma, epsilon)
            self.model = self.agent.model

    def get_action(self):
        q_values = self.model.predict(self.current_state[None])[0]
        prob = np.random.choice([True, False], p=[1-self.agent.epsilon, self.agent.epsilon])
        action = np.argmax(q_values) if prob else np.random.choice(self.action_space)
        return action

    def reset(self):
        if self.model.name == 'LSTM':
            self.state_iter, _ = create_sequences(self.df)
        else:
            self.state_iter, _ = states_iterator(self.df)
        self.current_state = next(self.state_iter)

        self.price_iter = close_price_iterator(self.df)
        self.current_price = next(self.price_iter)

        self.done = False

    def update_state(self):
        try:
            self.current_state = next(self.state_iter)
            self.current_price = next(self.price_iter)
        except StopIteration:
            self.current_state = np.zeros(self.state_dim)
            self.done = True

    def step(self, action):

        if action == 1:
            buy_count = (self.money * 0.3) / self.current_price
            self.actions[action](buy_count, self.current_price)
            reward = 0
            self.update_state()
        elif action == 2:
            profit = sum([self.current_price * v for v in self.orders.values()]) - \
                     sum([k * v for k, v in self.orders.items()])
            self.actions[action](self.current_price)
            reward = profit
            self.update_state()
        else:
            reward = 0
            self.update_state()

        return self.current_state, reward, self.done


class ReinforceEnvironment(Environment):

    def __init__(self, data_file, start_money, gamma):

        super().__init__(data_file, start_money)
        self.state_iter, self.ticks = states_iterator(self.df)
        self.current_state = next(self.state_iter)
        self.done = False
        self.state_dim = self.current_state.shape
        self.actions = {0: self.hold, 1: self.buy, 2: self.sell}
        self.action_space = len(self.actions)
        self.agent = Reinforce(self.action_space, self.state_dim, gamma)
        self.model = self.agent.model

    def reset(self):
        self.state_iter, _ = states_iterator(self.df)
        self.current_state = next(self.state_iter)

        self.price_iter = close_price_iterator(self.df)
        self.current_price = next(self.price_iter)

        self.done = False

    def update_state(self):
        try:
            self.current_state = next(self.state_iter)
            self.current_price = next(self.price_iter)
        except StopIteration:
            self.current_state = np.zeros(self.state_dim)
            self.done = True

    def step(self, action):

        if action == 1:
            buy_count = (self.money * 0.3) / self.current_price
            self.actions[action](buy_count, self.current_price)
            reward = 0
            self.update_state()
        elif action == 2:
            profit = sum([self.current_price * v for v in self.orders.values()]) - \
                     sum([k * v for k, v in self.orders.items()])
            self.actions[action](self.current_price)
            reward = profit
            self.update_state()
        else:
            reward = 0
            self.update_state()

        return self.current_state, reward, self.done
