# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/5/9

import numpy as np
import pandas as pd


class SarsaTable:
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
    def learn(self, s, a, r, s_, a_, alpha, g, tau):      # g为平均奖励率

        self.check_state_exist(s_)

        q_predict = self.q_table.ix[s, a]

        q_target = r - g * tau + self.q_table.ix[s_, a_]

        self.q_table.ix[s, a] = (1 - alpha) * q_predict + alpha * q_target

        return self.q_table
