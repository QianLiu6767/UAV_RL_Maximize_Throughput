# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/5/24

from Maze import Maze
from Brain import TD
import matplotlib.pyplot as plt
import numpy as np


def update():
    remark = []     # 记录平均奖励率
    remark_ = []    # 记录累计奖励
    global migration
    global average_migration
    migration = []
    average_migration = []

    cumulative_reward = 0
    g = 0  # Average return rate

    U = 0
    t = 0  # updates times

    global q_table

    for episode in range(100):

        observation = env.reset()

        while True:

            env.render()

            action = RL.choose_action(str(observation))       # 由m产生随机数，每步递加

            observation_, reward,  done, u = env.step(action)         # tau为step计数器

            q_table = RL.learn(str(observation), action, reward, str(observation_))

            t += 1                              # 更新次数加一

            U += u                              # 累计吞吐量
            cumulative_reward += reward         # 累计奖励

            if done:
                g = cumulative_reward / t  # 平均奖励率
                remark_.append(cumulative_reward)  # 记录累计奖励
                remark.append(g)  # 记录平均奖励率
                migration.append(U)
                average_migration.append(U / t)  # 记录平均吞吐量
                break

            observation = observation_

    print("Q-table:", q_table)



    """
    print('game over')
    env.destroy()
    """


if __name__ == "__main__":

    for i in range(10):
        env = Maze()
        RL = TD(actions=list(range(env.n_actions)))

        env.after(100, update)
        env.mainloop()

    plt.figure(1)
    plt.plot(np.arange(len(migration)), migration)
    plt.xlabel("Total Episodes")
    plt.ylabel("Total Migration")
    plt.figure(2)
    plt.plot(np.arange(len(average_migration)), average_migration)
    plt.xlabel("Total Episodes")
    plt.ylabel("Average_migration")
    plt.show()


