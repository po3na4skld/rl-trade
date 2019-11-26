import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def reward_graph(rewards, ep):
    y = rewards[0]
    for i in rewards[1:]:
        y = np.concatenate((y, i), axis=0)

    plt.figure(figsize=(20, 10))

    plt.plot(y, c='yellow')

    plt.title('Rewards on epoch #{}'.format(ep))
    plt.ylabel('Reward')

    plt.savefig('./graphs/rewards_epoch_{}.jpg'.format(ep))


def buy_sell_graph(buys, sells, price, ep):
    buys = np.asarray(buys).reshape(len(buys), )
    sells = np.asarray(sells).reshape(len(sells), )
    for i, j in zip(buys[1:], sells[1:]):
        buys[0] = np.concatenate((buys[0], i))
        sells[0] = np.concatenate((sells[0], j))

    buys = pd.DataFrame(buys[0])
    sells = pd.DataFrame(sells[0])

    plt.figure(figsize=(20, 10))

    plt.plot(price.iloc[:300], c='green')
    plt.scatter(x=buys[0].values, y=buys[1].values, c='red')
    plt.scatter(x=sells[0].values, y=sells[1].values, c='blue')

    plt.title('Buys and sells on epoch #{}'.format(ep))
    plt.legend(['Price', 'Buys', 'Sells'])

    plt.savefig('./graphs/buys_sells_graph_epoch_{}.jpg'.format(ep))
