from environments import QLearningEnvironment, ReinforceEnvironment
import numpy as np


def generate_session(env, agent):

    if agent == 'reinforce':
        states, actions, rewards = [], [], []
        env.reset()
        s = env.current_state
        for i in range(env.ticks):

            action_proba = env.agent.get_action_proba(s)
            a = np.random.choice(env.action_space, 1, p=action_proba)[0]
            next_s, r, done = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

            env.agent.train_step(states, actions, rewards)
            s = next_s

            if done:
                print(env)
                break

        return sum(rewards)
    else:
        env.reset()
        total_reward = 0
        s = env.current_state
        for t in range(env.ticks):
            a = env.get_action()
            next_s, r, done = env.step(a)
            env.agent.sess.run(env.agent.loss, {env.agent.states_ph: [s],
                                                env.agent.actions_ph: [a],
                                                env.agent.rewards_ph: [r],
                                                env.agent.next_states_ph: [next_s],
                                                env.agent.done_ph: [done]})
            total_reward += r
            s = next_s

            if done:
                print(env)
                break

        return total_reward


def run_train_session(epochs, agent, starting_money=20000, gamma=0.99, epsilon=0.5, save=True):

    data_file = 'stocks/train_data.csv'

    if agent == 'reinforce':
        env = ReinforceEnvironment(data_file, starting_money, gamma=gamma)
        model_name = 'reinforce.model'
    elif agent == 'q_learning_NN':
        env = QLearningEnvironment(data_file, starting_money, gamma=gamma, epsilon=epsilon, model_type=agent[:-2])
        model_name = agent + '.model'
    else:
        env = QLearningEnvironment(data_file, starting_money, gamma=gamma, epsilon=epsilon, model_type=agent[:-2])
        model_name = agent + '.model'

    for ep in range(1, epochs + 1):
        print('epoch: {} starts'.format(ep))
        rewards = [generate_session(env, agent) for _ in range(100)]
        env.agent.epsilon *= 0.99
        print('epoch {} ends.| mean reward: {}'.format(ep, np.mean(rewards)))

    if save:
        env.model.save('./models/' + model_name)


if __name__ == '__main__':
    agents = ['reinforce', 'q_learning_NN', 'q_learning_LSTM']
    run_train_session(10, agent=agents[1], starting_money=10000, save=True)
