# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/4/26

import numpy as np

import pandas as pd

# import json


class NTDTable:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table，其实也就是初始化为0
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    # 选动作

    def choose_action(self, observation, m):
        self.check_state_exist(observation)
        p0 = 0.01
        phi = 10e12
        u = (m ** 2) / (phi + m)
        pm = p0 / (1 + u)
        if np.random.uniform() < (1-pm):           # 选最优动作则flag=1
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
            flag = 1
        else:            # 随机选动作则flag=0
            action = np.random.choice(self.actions)
            flag = 0
        return action, flag

    # 学习过程

    def learn(self, S, A, R, s_, a_):

        c = 0

        n = len(R)

        for k in range(0, n):

            s = S[k]  # 需更新值的状态

            a = A[k]  # 需更新值的动作

            q_target = self.q_table.ix[s, a]

            G = 0

            for t in range(k, n):

                G += pow(self.gamma, t) * R[t]  # 累计奖励迭代增加G= R[] + R[] + R[]

            q_predict = G + pow(self.gamma, n) * int(self.q_table.iloc[s_, a_])

            error = q_predict - q_target  # 误差

            # 更新Q表

            self.q_table.ix[s, a] = q_target + self.lr * error

            c += error

        return self.q_table, c