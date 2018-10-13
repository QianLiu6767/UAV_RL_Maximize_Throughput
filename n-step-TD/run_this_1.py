# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/4/15

# 每一回合走n步，用n步累计奖励更新第一步的Q值

from maze_env import Maze

from RL_brain_1 import NTDTable

import numpy as np
import json
import matplotlib.pyplot as plt


def update():

    GAMMA = 0.9  # int型数据
    ALPHA = 0.1  # float型数据

    average_return = []  # 累计奖励
    step_size = []  # 记录每回合的步数

    G = 0  # 累计奖励初始化为0,int

    time = 0  # 追踪时间，第0步,int

    for episode in range(100):

        s = []                                                # S列表,list
        a = []                                                # A列表,list
        r = []                                                # R列表,list

        observation = env.reset()                             # 初始化状态,list

        env.render()  # 更新环境

        T = float('inf')                                      # float型数据

        while True:

            if time < T:

                s.append(observation)                                  # observation添加至表S中

                env.render()                                           # 更新环境

                q_table, eligibility_trace = RL.check_state_exist(str(observation))       # 更新Q表

                l = q_table.shape[0]                                   # 输出Q表的行数

                name = np.array(q_table.index)                          # 输出Q表行名并转换成list

                action = RL.choose_action(str(observation))            # t=0,根据S0选择A0,返回更新后的Q表

                a.append(action)                                       # 将action添加至A表中

                time += 1                                              # 下一步，第1步

                observation_, reward, done = env.step(action)          # t=1,根据A0得出S1、R0、done

                r.append(reward)                                       # 将reward添加至R表中

                if done:                                               # 如果下一个状态是终点（目标或障碍），则break，推出while循环，进入下一个episode

                    s.append(observation_)                             # observation添加至表S中

                    n = time                                           # 更新step_size

                    updatetime = n                                     # 更新时间tao

                    stateToUpdate = s[0]                             # 需更新值的状态

                    actionToUpdate = a[0]                            # 需更新值的动作

                    for t in range(0, n):

                        G += pow(GAMMA, t) * r[t]                    # 累计奖励迭代增加G= R[] + R[] + R[]

                    average_return.append(G / time)

                    step_size.append(n)

                    for t in range(0, n):

                        G += pow(GAMMA, t) * r[t]                    # 累计奖励迭代增加G= R[] + R[] + R[]

                    for i in range(l):                               # 找出最新状态动作值Q[S(n-1), A(n-1)]的Q表行数,返回索引值

                        # global q_predict

                        if json.loads(name[i]) == s[n - 1]:

                            # 实际值, q_predict = G + Q[S(n-1), A(n-1)]

                            q_predict = G + pow(GAMMA, n) * int(q_table.iloc[i, [a[n - 1]]])

                            for j in range(l):

                                if json.loads(name[j]) == stateToUpdate:

                                    # 要更新的状态动作值Q[S(UT), A(UT)]的估计值

                                    q_target = int(q_table.iloc[j, [actionToUpdate]])

                                    error = q_predict - q_target                                     # 误差

                                    # 更新

                                    # eligibility_trace.loc[j, [a[updatetime]]] += 1

                                    # 更新Q表

                                    q_table.iloc[j, [actionToUpdate]] = q_target + ALPHA * error      # * eligibility_trace



                    print('\n累计奖励：', G)
                    print('\nQ_table:\n', q_table)
                    print('\nepisode is :', episode)
                    print('\ntime is :', time)
                    print('\n                  ')

                    break

                else:
                    observation = observation_                          # 更新状态

    x = json.loads(str(step_size))
    y = json.loads(str(total_return))
    z = np.arange(0, 100, 1)

    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(x, y)
    ax1.set_xlabel('Step_Size')
    ax1.set_ylabel('Total Return')
    ax1.set_title('The relationship of total return and step_size')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(z, y)
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
