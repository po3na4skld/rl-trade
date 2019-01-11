from environments import QLearningEnvironment, ReinforceEnvironment
import numpy as np


def generate_session(env, alg_type, train=True):

    if alg_type == 'reinforce':
        states, actions, rewards = [], [], []
        env.reset()
        s = env.agent_input
        for i in range(env.ticks):
            action_probs = env.m.get_action_proba(s)

            a = np.random.choice(env.action_space, 1, p=action_probs)[0]

            next_s, r, done = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = next_s
            if done:
                print(env)
                break
        if train:
            env.m.train_step(states, actions, rewards)

        return sum(rewards)
    else:
        env.reset()
        total_reward = 0
        s = env.agent_input
        for t in range(env.ticks):
            a = env.get_action()
            next_s, r, done = env.step(a)
            if train:
                env.m_buy.sess.run(env.m_buy.loss, {env.m_buy.states_ph: [s],
                                                    env.m_buy.actions_ph: [a],
                                                    env.m_buy.rewards_ph: [r],
                                                    env.m_buy.next_states_ph: [next_s],
                                                    env.m_buy.is_done_ph: [done]})
            total_reward += r
            s = next_s
            if done:
                break
        return s, a, done, total_reward


def run_test_session(epochs, alg_type, save=True):
    data_file = 'stocks/dataset.csv'
    if alg_type == 'reinforce':
        env = ReinforceEnvironment(data_file, 2000, gamma=0.99)

        for ep in range(epochs):
            print('epoch: {} starts'.format(ep + 1))
            rewards = [generate_session(env, alg_type, train=True) for _ in range(100)]
            print('epoch {} ends | Mean reward: {}'.format(ep + 1, np.mean(rewards)))
        if save:
            env.m.model.save_weights('reinforce_weights.h5')
    else:
        env = QLearningEnvironment(20000, data_file, gamma=0.99, epsilon=0.01, model_type='NN')

        for ep in range(epochs):
            print('epoch: {} starts'.format(ep + 1))
            session = [generate_session(env, alg_type, train=True) for _ in range(10)]
            print('epoch {} ends | Total reward: {}'.format(ep + 1, session[-1][3]))
            print(env)
        if save:
            env.m.model.save_weights('q_learning_weights.h5')


if __name__ == '__main__':
    alg_types = ['reinforce',
                 'qlearning']
    run_test_session(20, alg_type=alg_types[0])
