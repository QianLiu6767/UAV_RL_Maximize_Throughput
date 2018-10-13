# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/4/16

# 每一回合走n步，用0-n的步累计奖励更新第0步的Q值，用1-n步的累计奖励更新第1步的Q值，用2-n的步累计奖励更新第2步的Q值，。。。，用n-1的步累计奖励更新第n-1步的Q值

from Maze import Maze

from RL_brain_1 import NTDTable

import numpy as np
import json
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import time as tm


def update():

    GAMMA = 0.9  # int型数据
    ALPHA = 1  # float型数据

    average_return = []  # 累计奖励
    step_size = []  # 记录每回合的步数
    turn = 200

    time = 0  # 追踪时间，第0步,int
    G = 0  # 累计奖励初始化为0,int

    for episode in range(turn):

        t = 0

        s = []  # S列表,list
        a = []  # A列表,list
        r = []  # R列表,list

        observation = env.reset_uav()  # 初始化状态,list
        env.render()

        T = float('inf')  # float型数据

        while True:

            if t < T:

                s.append(observation)  # observation添加至表S中

                env.render()  # 更新环境

                q_table, eligibility_trace = RL.check_state_exist(str(observation))  # 更新Q表

                l = q_table.shape[0]  # 输出Q表的行数

                name = np.array(q_table.index)  # 输出Q表行名并转换成list

                action = RL.choose_action(str(observation))  # t=0,根据S0选择A0,返回更新后的Q表

                a.append(action)  # 将action添加至A表中

                time += 1  # 下一步，第1步

                t += 1

                observation_, reward, done = env.step(action)  # t=1,根据A0得出S1、R0、done

                r.append(reward)  # 将reward添加至R表中

                if done:  # 如果下一个状态是终点（目标或障碍），则break，推出while循环，进入下一个episode

                    s.append(observation_)  # observation添加至表S中

                    n = time   # 总步数

                    T = t        # 更新step_size

                    for t in range(0, T):            # 计算一个回合累计奖励
                        # G += pow(GAMMA, t) * r[t]  # 累计奖励迭代增加G= R[] + R[] + R[]
                        G += r[t]

                    average_return.append(G / n)  # 平均奖励

                    step_size.append(T)

                    for k in range(0, T):

                        stateToUpdate = s[k]   # 需更新值的状态

                        actionToUpdate = a[k]  # 需更新值的动作

                        for i in range(l):  # 找出最新状态动作值Q[S(n-1), A(n-1)]的Q表行数,返回索引值

                            # global q_predict

                            if json.loads(name[i]) == s[T - 1]:

                                # 实际值, q_predict = G + Q[S(n-1), A(n-1)]

                                q_predict = G + pow(GAMMA, T) * int(q_table.iloc[i, [a[T - 1]]])

                                for j in range(l):

                                    if json.loads(name[j]) == stateToUpdate:
                                        # 要更新的状态动作值Q[S(UT), A(UT)]的估计值

                                        q_target = int(q_table.iloc[j, [actionToUpdate]])

                                        error = q_predict - q_target  # 误差

                                        # 更新

                                        # eligibility_trace.loc[j, [a[updatetime]]] += 1

                                        # 更新Q表

                                        q_table.iloc[j, [actionToUpdate]] = q_target + ALPHA * error  # * eligibility_trace

                    print('\n累计奖励：', G)
                    # print('\nQ_table:\n', q_table)
                    print('\nepisode is :', episode)
                    print('\nstep is :', T)
                    print('\n                  ')

                    break

                else:
                    observation = observation_  # 更新状态

    observation = env.reset_uav()

    env.render()

    x = json.loads(str(step_size))

    y = np.arange(0, turn, 1)

    z = json.loads(str(average_return))

    """
    fig = plt.figure(figsize=(15, 15))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.scatter(x, z)
    ax1.set_xlabel('Step_Size')
    ax1.set_ylabel('Total Return')
    ax1.set_title('The relationship of total return and step_size')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.scatter(y, z)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Return')
    ax2.set_title('The relationship of total return and episode')

    figure = fig.add_subplot(3, 1, 3)
    ax3 = Axes3D(figure)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Step_Size')
    ax3.set_zlabel('Total Return')
    ax3.plot_surface(y, x, z)

    plt.show()

    """
    """
    plt.figure(1)
    plt.scatter(x, z, c='red')
    plt.xlabel('Step_Size')
    plt.ylabel('Total Return')

    plt.figure(3)
    plt.plot(y, x, c='green')
    plt.xlabel('Episode')
    plt.ylabel('Step_Size')
    
    """

    plt.figure(2)
    plt.plot(y, z, c='blue')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')



    """
    figure = plt.figure(4)
    ax = Axes3D(figure)
    X = np.arange(0, turn, 1)
    Y = json.loads(str(step_size))
    Z = json.loads(str(total_return))

    ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Step_Size')
    ax.set_zlabel('Total Return')
    """

    plt.show()

    # end of game

    print('game over')

    env.destroy()


if __name__ == "__main__":
    env = Maze()

    RL = NTDTable(actions=list(range(env.n_actions)))

    env.after(100, update)

    env.mainloop()
