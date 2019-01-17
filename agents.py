from keras.layers import Dense, Activation, InputLayer, Dropout, CuDNNLSTM, BatchNormalization, Input
from keras.models import Sequential, Model
from collections import deque
import tensorflow as tf
import functools


def property_with_check(input_fn):
    attribute = '_cache_' + input_fn.__name__

    @property
    @functools.wraps(input_fn)
    def check_attr(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, input_fn(self))
        return getattr(self, attribute)

    return check_attr


class Reinforce:

    def __init__(self, action_space, state_dim, gamma, lr=1e-4):
        self.n_actions = action_space
        self.state_dim = state_dim
        self.gamma = gamma
        self.lr = lr
        self.sess = tf.Session()

        self._policy = None
        self._log_policy = None
        self.states_ph = tf.placeholder(tf.float32, shape=(None,) + self.state_dim)
        self.actions_ph = tf.placeholder(tf.int32, shape=[None])
        self.cumulative_rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self.done_ph = tf.placeholder(tf.bool, shape=[None])

        self.model = self._build_model()
        self._loss = None
        self._optimizer = None
        self._all_weights = None
        self.get_action_proba = lambda s: self.policy.eval({self.states_ph: [s]}, session=self.sess)[0]
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_model(self):
        model = Sequential(name='NN')
        model.add(InputLayer(self.state_dim))

        model.add(Dense(100))
        model.add(Dropout(0.25))
        model.add(Activation('relu'))

        model.add(Dense(100))
        model.add(Dropout(0.25))
        model.add(Activation('relu'))

        model.add(Dense(self.n_actions))
        model.add(Activation('linear'))

        return model

    @property_with_check
    def policy(self):
        logits = self.model(self.states_ph)
        self._policy = tf.nn.softmax(logits)
        return self._policy

    @property_with_check
    def log_policy(self):
        logits = self.model(self.states_ph)
        self._log_policy = tf.nn.log_softmax(logits)
        return self._log_policy

    @property_with_check
    def all_weights(self):
        self._all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return self._all_weights

    @property_with_check
    def loss(self):
        indices = tf.stack([tf.range(tf.shape(self.log_policy)[0]), self.actions_ph], axis=-1)
        log_policy_for_actions = tf.gather_nd(self.log_policy, indices)
        loss = tf.reduce_mean(log_policy_for_actions * self.cumulative_rewards_ph)
        entropy = -tf.reduce_mean(self.policy * self.log_policy, 1)

        self._loss = -loss - 0.1 * entropy
        return self._loss

    @property_with_check
    def optimizer(self):
        opt = tf.train.AdamOptimizer(self.lr)
        opt.minimize(self.loss, var_list=self.all_weights)
        self._optimizer = opt
        return self._optimizer

    def get_cumulative_rewards(self, rewards):
        cumulative_rewards = deque([rewards[-1]])
        for i in range(len(rewards) - 2, -1, -1):
            cumulative_rewards.appendleft(rewards[i] + self.gamma * cumulative_rewards[0])
        return cumulative_rewards

    def train_step(self, _states, _actions, _rewards):
        _cumulative_rewards = self.get_cumulative_rewards(_rewards)
        self.sess.run(self.loss, {self.states_ph: _states,
                                  self.actions_ph: _actions,
                                  self.cumulative_rewards_ph: _cumulative_rewards})

    def get_summary(self):
        self.model.summary()


class QLearningNN:

    def __init__(self, action_space, state_dim, gamma, epsilon, lr=1e-4):
        self.n_actions = action_space
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.sess = tf.Session()
        self.model = self._build_model()
        self.model.name = 'NN'

        self.states_ph = tf.placeholder(tf.float32, shape=(None, ) + self.state_dim)
        self.actions_ph = tf.placeholder(tf.int32, shape=[None])
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self.next_states_ph = tf.placeholder(tf.float32, shape=(None, ) + self.state_dim)
        self.done_ph = tf.placeholder(tf.bool, shape=[None])

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self._loss = None
        self._optimizer = None

    def _build_model(self):
        model = Sequential()
        model.add(InputLayer(self.state_dim))

        model.add(Dense(100))
        model.add(Dropout(0.25))
        model.add(Activation('relu'))

        model.add(Dense(100))
        model.add(Dropout(0.25))
        model.add(Activation('relu'))

        model.add(Dense(self.n_actions))
        model.add(Activation('linear'))

        return model

    def get_summary(self):
        self.model.summary()

    @property_with_check
    def loss(self):
        one_hot_action = tf.one_hot(self.actions_ph, self.n_actions)
        q_value = tf.reduce_sum(self.model(self.states_ph) * one_hot_action, axis=1)
        next_q_value = tf.reduce_sum(self.model(self.next_states_ph) * one_hot_action, axis=1)
        target = self.rewards_ph + self.gamma * tf.reduce_max(next_q_value)
        loss = tf.reduce_mean(tf.square(target - q_value))
        self._loss = loss
        return self._loss

    @property_with_check
    def optimizer(self):
        opt = tf.train.AdamOptimizer(self.lr)
        opt.minimize(self.loss)
        self._optimizer = opt
        return self._optimizer


class QLearningLSTM:

    def __init__(self, action_space, state_dim, gamma, epsilon, lr=1e-4):
        self.n_actions = action_space
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.sess = tf.Session()
        self.model = self._build_model()
        self.model.name = 'LSTM'

        self.states_ph = tf.placeholder(tf.float32, shape=(None, ) + self.state_dim)
        self.actions_ph = tf.placeholder(tf.int32, shape=[None])
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self.next_states_ph = tf.placeholder(tf.float32, shape=(None, ) + self.state_dim)
        self.done_ph = tf.placeholder(tf.bool, shape=[None])

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self._loss = None
        self._optimizer = None

    def _build_model(self):

        inputs = Input(self.state_dim)

        line_one = CuDNNLSTM(128, return_sequences=True)(inputs)
        line_one = BatchNormalization()(line_one)

        line_one = CuDNNLSTM(128, return_sequences=True)(line_one)
        line_one = BatchNormalization()(line_one)

        line_one = CuDNNLSTM(128)(line_one)
        line_one = BatchNormalization()(line_one)

        con = Dense(100)(line_one)
        con = Dropout(0.25)(con)
        con = Activation('relu')(con)

        con = Dense(self.n_actions)(con)
        output = Activation('linear')(con)

        model = Model(inputs=inputs, outputs=output)

        return model

    def get_summary(self):
        self.model.summary()

    @property_with_check
    def loss(self):
        one_hot_action = tf.one_hot(self.actions_ph, self.n_actions)
        q_value = tf.reduce_sum(self.model(self.states_ph) * one_hot_action, axis=1)
        next_q_value = tf.reduce_sum(self.model(self.next_states_ph) * one_hot_action, axis=1)
        target = self.rewards_ph + self.gamma * tf.reduce_max(next_q_value)
        loss = tf.reduce_mean(tf.square(target - q_value))
        self._loss = loss
        return self._loss

    @property_with_check
    def optimizer(self):
        opt = tf.train.AdamOptimizer(self.lr)
        opt.minimize(self.loss)
        self._optimizer = opt
        return self._optimizer
