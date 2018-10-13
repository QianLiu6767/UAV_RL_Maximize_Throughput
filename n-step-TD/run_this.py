# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/4/26

from Maze import Maze

from RL_brain import RL

import numpy as np
import json
import matplotlib.pyplot as plt

turn = 100


def update():

    for episode in range(turn):

        # N = 0
        m = 0
        t = 0

        total_return = []    # 累计奖励
        average_return = []  # 平均奖励
        c = []               # 误差

        # step_size = []  # 记录每回合的步数

        s = []   # 状态集
        a = []   # 动作集
        r = []   # 奖励集

        cumulative_reward = 0

        observation = env.reset_uav()

        # env.render()

        while True:

            env.render()

            s.append(observation)  # observation添加至表S中

            action, flag = RL.choose_action(str(observation), m)  # 由m产生随机数，每步递加

            a.append(action)  # 将action添加至A表中

            observation_, reward, tau, done = env.step(action)  # tau为step计数器

            s.append(observation_)  # observation添加至表S中

            r.append(reward)  # 将reward添加至R表中

            if flag:        # 如果动作不是随机选的

                # N += 1      # 更新次数加一

                cumulative_reward += reward  # 累计奖励

                t = t + tau                  # step次数加一

                g = cumulative_reward / t    # 平均奖励率

                total_return.append(cumulative_reward)  # 记录累计奖励

                average_return.append(g)  # 记录平均奖励率

            if done:       # 如果下一个状态是终点（目标或障碍），则break，推出while循环，进入下一个episode

                Q_tabel, cost = RL.learn(s, a, r, observation, action)
                c.append(cost)

                print('\nepisode is :', episode)
                print('\nThe number of unrandom step is :', t)
                print('\n累计奖励：', cumulative_reward)
                print('\nQ_table:\n', Q_tabel)
                print('\n                  ')

                break
            observation = observation_

            m += 1
    """
    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(np.arange(0, turn, 1), average_return)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cost')
    ax1.set_title('The relationship of total return and step_size')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(np.arange(0, turn, 1), c)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Return')
    ax2.set_title('The relationship of total return and episode')

    plt.show()
    """

    # end of game

    print('game over')

    env.destroy()


if __name__ == "__main__":

    env = Maze()

    RL = RL(actions=list(range(env.n_actions)))

    env.after(100, update)

    env.mainloop()
