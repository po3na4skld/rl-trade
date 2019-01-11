from agents import QLearningLSTM, QLearningNN, Reinforce
from signal_generation import generate_signals
from data_handler import *
import numpy as np


class QLearningEnvironment:

    def __init__(self, data_file, start_money, gamma, epsilon, model_type):

        # portfolio
        self.money_at_start = start_money
        self.money = start_money
        self.currency_count = 0
        self.buy_price = 0
        self.portfolio = np.asarray([self.money, self.currency_count])
        self.buy_mode = 1

        # data variables
        self.df = preprocess_data(data_file)

        # data iterators
        self.price_iter = create_price_frame(self.df)
        self.current_price = next(self.price_iter)
        self.data_iter, self.ticks = create_sequences(self.df)
        self.current_data = next(self.data_iter)

        # agent variables
        self.done = False
        self.agent_input = generate_signals(self.current_data, self.buy_mode)
        self.state_dim = self.agent_input.shape
        self.actions = {0: self.hold, 1: self.buy, 2: self.sell}
        self.action_space = len(self.actions)
        if model_type == 'NN':
            self.m = QLearningNN(self.action_space, self.state_dim, gamma, epsilon)
        else:
            self.m = QLearningLSTM(self.action_space, self.state_dim, self.portfolio.shape, gamma, epsilon)

    def reset(self):
        self.portfolio = np.asarray([self.money_at_start, 0])

        self.data_iter, _ = create_sequences(self.df)
        self.current_data = next(self.data_iter)

        self.price_iter = create_price_frame(self.df)
        self.current_price = next(self.price_iter)

        self.done = False

    def update_state(self):
        try:
            self.portfolio = np.asarray([self.money, self.currency_count])
            self.current_data = next(self.data_iter)
            self.current_price = next(self.price_iter)
            self.agent_input = generate_signals(self.current_data, self.buy_mode)
        except StopIteration:
            self.current_data = np.zeros((10,))
            self.done = True

    def step(self, action):
        if self.currency_count > 0:
            if action == 1 and self.buy_mode == 0:
                reward = -1
                self.update_state()
            elif action == 2:
                self.actions[action](self.current_price)
                reward = (self.current_price - self.buy_price) * 2
                self.buy_mode = 1
                self.update_state()
            else:
                reward = 0
                self.update_state()
        else:
            if action == 1:
                buy_count = (self.money * 0.3) / self.current_price
                self.actions[action](buy_count, self.current_price)
                reward = 0
                self.buy_mode = 0
                self.update_state()
            elif action == 2 and self.buy_mode == 1:
                reward = -1
                self.update_state()
            else:
                reward = 0
                self.update_state()

        return self.agent_input, reward, self.done

    def get_action(self):
        if self.m.model.name == 'QlearningLSTM':
            q_values = self.m.model.predict([self.current_data[None], self.portfolio[None]])[0]
        else:

            q_values = self.m.model.predict(self.agent_input[None])[0]

        prob = np.random.choice([True, False], p=[1-self.m.epsilon, self.m.epsilon])
        if prob:
            action = np.argmax(q_values)
        else:
            action = np.random.choice(self.action_space)

        return action

    def buy(self, count, price):
        if self.money < count * price:
            pass
        else:
            self.currency_count += count
            self.money = self.money - count * price
            self.buy_price = price

    def sell(self, price):
        self.money += self.currency_count * price
        self.currency_count = 0

    def hold(self):
        pass

    def __repr__(self):
        return 'Total profit: {:.2f} '.format(self.money - self.money_at_start)


class ReinforceEnvironment:

    def __init__(self, data_file, start_money, gamma):

        # portfolio
        self.money_at_start = start_money
        self.money = start_money
        self.currency_count = 0
        self.buy_price = 0
        self.portfolio = np.asarray([self.money, self.currency_count])
        self.buy_mode = 1

        # data variables
        self.df = preprocess_data(data_file)

        # data iterators
        self.price_iter = create_price_frame(self.df)
        self.current_price = next(self.price_iter)
        self.data_iter, self.ticks = create_sequences(self.df)
        self.current_data = next(self.data_iter)

        # agent variables
        self.done = False
        self.agent_input = generate_signals(self.current_data, self.buy_mode)
        self.state_dim = self.agent_input.shape
        self.actions = {0: self.hold, 1: self.buy, 2: self.sell}
        self.action_space = len(self.actions)
        self.m = Reinforce(self.action_space, self.state_dim, gamma)

    def reset(self):
        self.portfolio = np.asarray([self.money_at_start, 0])

        self.data_iter, _ = create_sequences(self.df)
        self.current_data = next(self.data_iter)

        self.price_iter = create_price_frame(self.df)
        self.current_price = next(self.price_iter)

        self.done = False

    def update_state(self):
        try:
            self.portfolio = np.asarray([self.money, self.currency_count])
            self.current_data = next(self.data_iter)
            self.current_price = next(self.price_iter)
            self.agent_input = generate_signals(self.current_data, self.buy_mode)
        except StopIteration:
            self.current_data = np.zeros((10,))
            self.done = True

    def step(self, action):
        if self.currency_count > 0:
            if action == 1 and self.buy_mode == 0:
                reward = -1
                self.update_state()
            elif action == 2:
                self.actions[action](self.current_price)
                reward = (self.current_price - self.buy_price) * 2
                self.buy_mode = 1
                self.update_state()
            else:
                reward = 0
                self.update_state()
        else:
            if action == 1:
                buy_count = (self.money * 0.3) / self.current_price
                self.actions[action](buy_count, self.current_price)
                reward = 0
                self.buy_mode = 0
                self.update_state()
            elif action == 2 and self.buy_mode == 1:
                reward = -1
                self.update_state()
            else:
                reward = 0
                self.update_state()

        return self.agent_input, reward, self.done

    def buy(self, count, price):
        if self.money < count * price:
            pass
        else:
            self.currency_count += count
            self.money = self.money - count * price
            self.buy_price = price

    def sell(self, price):
        self.money += self.currency_count * price
        self.currency_count = 0

    def hold(self):
        pass

    def __repr__(self):
        return 'Total profit: {:.2f} '.format(self.money - self.money_at_start)
