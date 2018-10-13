# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/4/15

# n-step 变形体的brain，无神经网络

import numpy as np

import pandas as pd


class RL(object):

    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):

        self.actions = action_space  # a list

        self.lr = learning_rate

        self.gamma = reward_decay

        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):

        if state not in self.q_table.index:

            # append new state to q table

            self.q_table = self.q_table.append(

                pd.Series(

                    [0]*len(self.actions),

                    index=self.q_table.columns,

                    name=state,

                )

            )
        else:
            self.q_table = self.q_table

        return self.q_table

    def choose_action(self, observation):

        self.check_state_exist(observation)

        # action selection

        if np.random.rand() < self.epsilon:

            # choose best action

            state_action = self.q_table.loc[observation, :]

            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value

            action = state_action.idxmax()

        else:

            # choose random action

            action = np.random.choice(self.actions)

        return action


class NTDTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):

        super(NTDTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.

        self.lambda_ = trace_decay                     # lambda值

        self.eligibility_trace = self.q_table.copy()   # 类似于Q表，状态动作表，经历一次就对应值+1

    def check_state_exist(self, state):

        if state not in self.q_table.index:

            # append new state to q table

            to_be_append = pd.Series(

                    [0] * len(self.actions),

                    index=self.q_table.columns,

                    name=state,

                )

            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)    # 同时更新

        return self.q_table, self.eligibility_trace

    def choose_action(self, observation):

        self.check_state_exist(observation)

        # action selection

        if np.random.rand() < self.epsilon:

            # choose best action

            state_action = self.q_table.loc[observation, :]

            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value

            action = state_action.idxmax()

        else:

            # choose random action

            action = np.random.choice(self.actions)

        return action
