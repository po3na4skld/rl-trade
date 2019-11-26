from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

from keras.optimizers import Adam

from agents import Agent


class SarsaAgent(Agent):

    def __init__(self, state_dim, action_space, epsilon, gamma, lr):
        self._model = self._get_model(state_dim, action_space)
        self.agent = SARSAAgent(self._model, nb_actions=action_space, gamma=gamma,
                                policy=EpsGreedyQPolicy(epsilon), test_policy=EpsGreedyQPolicy(eps=0.01))

        self.agent.compile(Adam(lr))

    def model_summary(self):
        print(self._model.summary())
