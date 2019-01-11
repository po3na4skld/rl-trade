import matplotlib.pyplot as plt


def reward_graph(rewards):
    plt.plot(rewards, c='yellow')
    plt.title('Rewards')
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.style.use('fivethirtyeight')
    plt.show()
