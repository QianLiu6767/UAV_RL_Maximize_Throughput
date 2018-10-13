# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/5/18

from Maze import Maze

from RL_brain import NTDTable

import numpy as np
import json
import matplotlib.pyplot as plt


total_return = []       # 累计奖励
step_size = []          # 记录每回合的步数
c = []
turn = 100


def update():

    for episode in range(turn):

        global q_table

        s = []
        a = []
        r = []
        time = 0

        G = 0

        observation = env.reset_uav()

        while True:

            env.render()

            s.append(observation)  # observation添加至表S中

            action = RL.choose_action(str(observation))

            a.append(action)  # 将action添加至A表中

            time += 1

            observation_, reward, done = env.step(action)

            s.append(observation_)  # observation添加至表S中

            r.append(reward)  # 将reward添加至R表中

            G += reward

            observation = observation_  # 更新状态

            if done:       # 如果下一个状态是终点（目标或障碍），则break，推出while循环，进入下一个episode

                total_return.append(G)

                step_size.append(time)

                q_table, cost = RL.learn(s, a, r, time)

                c.append(cost)

                print('\n累计奖励：', G)
                print('\nQ_table:\n', q_table)
                print('\nepisode is :', episode)
                print('\ntime is :', time)
                print('\n                  ')

                break

    x = json.loads(str(step_size))
    y = json.loads(str(total_return))
    z = json.loads(str(c))

    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(np.arange(0, turn, 1), z)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cost')
    ax1.set_title('The relationship of total return and step_size')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(np.arange(0, turn, 1), y)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Return')
    ax2.set_title('The relationship of total return and episode')

    plt.show()
    # end of game

    print('game over')

    env.destroy()


if __name__ == "__main__":

    env = Maze()

    RL = NTDTable(actions=list(range(env.n_actions)))

    env.after(100, update)

    env.mainloop()
