# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/4/23

from Maze import Maze
from RL_brain import NTDTable

import numpy as np
import json
import matplotlib.pyplot as plt

GAMMA = 1    # int型数据
ALPHA = 0.1    # float型数据

total_return = []  # 累计奖励
step_size = []  # 记录每回合的步数

turn = 100


def update():

    for episode in range(turn):

        s = []  # S列表,list
        a = []  # A列表,list
        r = []  # R列表,list

        observation = env.reset_uav()  # 初始化无人机状态,list
        env.render()

        time = 0  # 追踪时间，第0步,int

        T = float('inf')  # float型数据

        while True:

            if time < T:

                s.append(observation)  # observation添加至表S中

                env.render()  # 更新环境

                q_table = RL.check_state_exist(str(observation))  # 更新Q表

                l = q_table.shape[0]  # 输出Q表的行数

                name = np.array(q_table.index)  # 输出Q表行名并转换成list

                action = RL.choose_action(str(observation))  # t=0,根据S0选择A0,返回更新后的Q表

                a.append(action)  # 将action添加至A表中

                time += 1  # 下一步，第1步

                observation_, reward, done = env.step(action)  # t=1,根据A0得出S1、R0、done

                r.append(reward)  # 将reward添加至R表中

                if done:  # 如果下一个状态是终点（目标或障碍），则break，推出while循环，进入下一个episode

                    s.append(observation_)  # observation添加至表S中

                    n = time  # 更新step_size

                    G = 0     # 累计奖励初始化为0,int

                    for t in range(0, n):              # 计算一个回合累计奖励,G += pow(GAMMA, t) * r[t]

                        G += r[t]

                    total_return.append(G)  # 累计奖励list

                    step_size.append(n)         # 步长list

                    print('\n累计奖励：', G)
                    print('\nepisode is :', episode)
                    print('\nstep is :', n)
                    print('\n               ')

                    if episode == 0:

                        print("\n状态集：", s)
                        print("\n动作集：", a)
                        print("\n奖励集：", r)

                        h = s[:]
                        m = a[:]
                        o = r[:]

                        for k in range(0, n):

                            stateToUpdate = s[k]  # 需更新值的状态

                            actionToUpdate = a[k]  # 需更新值的动作

                            G = 0     # 累计奖励初始化为0,int

                            for t in range(k, n):

                                G += pow(GAMMA, t) * r[t]  # 累计奖励迭代增加G= R[] + R[] + R[]

                            for i in range(l):  # 找出最新状态动作值Q[S(n-1), A(n-1)]的Q表行数,返回索引值

                                # global q_predict

                                if json.loads(name[i]) == s[n - 1]:

                                    # 实际值, q_predict = G + Q[S(n-1), A(n-1)]

                                    q_predict = G + pow(GAMMA, n) * int(q_table.iloc[i, [a[n - 1]]])

                                    for j in range(l):

                                        if json.loads(name[j]) == stateToUpdate:

                                            # 要更新的状态动作值Q[S(UT), A(UT)]的估计值

                                            q_target = int(q_table.iloc[j, [actionToUpdate]])

                                            error = q_predict - q_target  # 误差

                                            # 更新Q表

                                            q_table.iloc[j, [actionToUpdate]] = q_target + ALPHA * error  # * eligibility_trace

                    else:

                        print("\n状态集：", s)
                        print("\n动作集：", a)
                        print("\n奖励集：", r)

                        if G > max(total_return):   # 本回合累计平均奖励大于list中最大值，更新Q表

                            h = s[:]
                            m = a[:]
                            o = r[:]

                            for k in range(0, n):

                                stateToUpdate = s[k]   # 需更新值的状态

                                actionToUpdate = a[k]  # 需更新值的动作

                                G = 0  # 累计奖励初始化为0,int

                                for t in range(k, n):

                                    G += pow(GAMMA, t) * r[t]  # 累计奖励迭代增加G= R[] + R[] + R[]

                                for i in range(l):  # 找出最新状态动作值Q[S(n-1), A(n-1)]的Q表行数,返回索引值

                                    # global q_predict

                                    if json.loads(name[i]) == s[n - 1]:

                                        # 实际值, q_predict = G + Q[S(n-1), A(n-1)]

                                        q_predict = G + pow(GAMMA, n) * int(q_table.iloc[i, [a[n - 1]]])

                                        for j in range(l):

                                            if json.loads(name[j]) == stateToUpdate:

                                                # 要更新的状态动作值Q[S(UT), A(UT)]的估计值

                                                q_target = int(q_table.iloc[j, [actionToUpdate]])

                                                error = q_predict - q_target  # 误差

                                                # 更新Q表

                                                q_table.iloc[j, [actionToUpdate]] = q_target + ALPHA * error

                        """
                        else:

                            s = []
                            a = []
                            r = []
                            n = 0

                            s = h[:]
                            a = m[:]
                            r = o[:]

                            n = len(r)

                            for k in range(0, n):

                                stateToUpdate = s[k]   # 需更新值的状态

                                actionToUpdate = a[k]  # 需更新值的动作

                                G = 0  # 累计奖励初始化为0,int

                                for t in range(k, n):

                                    G += pow(GAMMA, t) * r[t]  # 累计奖励迭代增加G= R[] + R[] + R[]

                                for i in range(l):  # 找出最新状态动作值Q[S(n-1), A(n-1)]的Q表行数,返回索引值

                                    # global q_predict

                                    if json.loads(name[i]) == s[n - 1]:

                                        # 实际值, q_predict = G + Q[S(n-1), A(n-1)]

                                        q_predict = G + pow(GAMMA, n) * int(q_table.iloc[i, [a[n - 1]]])

                                        for j in range(l):

                                            if json.loads(name[j]) == stateToUpdate:

                                                # 要更新的状态动作值Q[S(UT), A(UT)]的估计值

                                                q_target = int(q_table.iloc[j, [actionToUpdate]])

                                                error = q_predict - q_target  # 误差

                                                # 更新Q表

                                                q_table.iloc[j, [actionToUpdate]] = q_target + ALPHA * error"""

                    print("\n最优状态集", h)
                    print("\n最优动作集", m)
                    print("\n最优奖励集", o)
                    break

                else:

                    observation = observation_  # 更新状态

    observation = env.reset_uav()

    env.render()

    x = json.loads(str(step_size))

    y = np.arange(0, turn, 1)

    z = json.loads(str(total_return))

    """plt.figure(1)
    plt.plot(x, z)
    plt.xlabel('Step_Size')
    plt.ylabel('Total Return')"""

    plt.figure(2)
    plt.plot(y, z)
    plt.xlabel('Episode')
    plt.ylabel('Total Return')

    plt.figure(3)
    plt.plot(y, x)
    plt.xlabel('Episode')
    plt.ylabel('Step_Size')

    plt.show()


    # end of game

    print('game over')

    env.destroy()


if __name__ == "__main__":
    env = Maze()

    RL = NTDTable(actions=list(range(env.n_actions)))

    env.after(100, update)

    env.mainloop()
