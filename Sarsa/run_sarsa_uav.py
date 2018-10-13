# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/5/9

from ENV import Maze
from brain import SarsaTable
import matplotlib.pyplot as plt
import numpy as np


def update():
    m = 0

    t = 0
    N = 0  # updates times

    cumulative_reward = 0
    g = 0  # Average return rate

    remark_ = []
    remark = []

    for episode in range(100):

        observation = env.reset()

        action, flag = RL.choose_action(str(observation), m)

        while True:

            env.render()

            observation_, reward, tau, done = env.step(action)     # tau为step计数器

            action_, flag = RL.choose_action(str(observation_), m)

            alpha = 1 / (1 + N)

            RL.learn(str(observation), action, reward, str(observation_), action_, alpha, g, tau)

            if flag:
                N += 1

                cumulative_reward += reward         # 累计奖励
                t = t + tau
                g = cumulative_reward / t           # 平均奖励率

                remark_.append(cumulative_reward)   # 记录累计奖励
                remark.append(g)  # 记录平均奖励率

            observation = observation_
            action = action_

            m += 1

            if done:
                break

    plt.figure(1)
    plt.plot(np.arange(len(remark)), remark)
    plt.xlabel("total steps")
    plt.ylabel("Average return rate")
    plt.figure(2)
    plt.plot(np.arange(len(remark_)), remark_)
    plt.xlabel("total steps")
    plt.ylabel("cumulative_reward")
    plt.show()
    print('game over')
    env.destroy()


if __name__ == "__main__":

    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
