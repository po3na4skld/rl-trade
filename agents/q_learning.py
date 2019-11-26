from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy

from keras.optimizers import Adam

from agents import Agent


class QLearningAgent(Agent):

    def __init__(self, state_dim, action_space, epsilon, lr):
        self._model = self._get_model(state_dim, action_space)
        self.agent = DQNAgent(self._model, policy=EpsGreedyQPolicy(epsilon), test_policy=EpsGreedyQPolicy(eps=0.01))

        self.agent.compile(Adam(lr))

    def model_summary(self):
        print(self._model.summary())
