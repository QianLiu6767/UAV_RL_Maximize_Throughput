# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/7/31

# 路径规划

import numpy as np
import tkinter as tk
import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pyttsx3

UNIT = 20
MAZE_H = 20
MAZE_W = 20

np.random.seed(1)
tf.set_random_seed(1)


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                             '11', '12', '13', '14', '15']
        self.n_actions = len(self.action_space)
        self.n_features = 3  # 无人机位置, 电池量，信道
        self.title('UAV测试环境-Q')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # --------------------------构建固定点---------------------------
        self.oval_center = np.array([[0, 0], [50, 70], [80, 45], [100, 100], [100, 125], [130, 110], [150, 170], [200, 200],
                                     [220, 230], [250, 260], [280, 300], [300, 300], [355, 370], [380, 345], [400, 400]])
        for i in range(15):
            self.canvas.create_oval(self.oval_center[i, 0] - 10, self.oval_center[i, 1] - 10,
                                    self.oval_center[i, 0] + 10, self.oval_center[i, 1] + 10,
                                    fill='blue')

        # --------------------------用户--------------------------------
        self.user_center = np.array([[50, 30], [150, 125], [200, 170], [310, 270], [380, 380]])
        for i in range(5):
            self.canvas.create_oval(self.user_center[i, 0] - 5, self.user_center[i, 1] - 5,
                                    self.user_center[i, 0] + 5, self.user_center[i, 1] + 5,
                                    fill='black')

        # --------------------------UAV标识-----------------------------
        self.img = tk.PhotoImage(file="UAV.png")
        self.uav = self.canvas.create_image((5, 5), image=self.img)

        # pack all
        self.canvas.pack()

    def reset_uav(self):
        self.update()
        time.sleep(0.1)
        self.battery = 10
        self.canvas.delete(self.uav)
        self.uav = self.canvas.create_image((5, 5), image=self.img)

        return np.hstack((np.array([self.canvas.coords(self.uav)[0],
                                    self.canvas.coords(self.uav)[1], self.battery])))

    def step(self, action):
        s = np.array(self.canvas.coords(self.uav))
        for i in range(15):
            if action == i:
                self.canvas.delete(self.uav)
                point = self.oval_center[i, :]
                self.uav = self.canvas.create_image((point[0], point[1]), image=self.img)
                break

        next_coords = self.canvas.coords(self.uav)  # next state,返回值是列表[ , ]

        u = [10, 10, 10, 10, 10]

        # reward function
        for j in range(15):
            if next_coords == self.oval_center[j, :].tolist():
                p = 0.1
                rho = 10e-5
                sigma = 9.999999999999987e-15
                # 飞行能耗，归一化问题
                ef = p * (np.sqrt((s[0] - self.oval_center[j, 0]) ** 2 +
                                  (s[1] - self.oval_center[j, 1]) ** 2)) / 80

                # 盘旋能耗
                distance = np.array([0, 0, 0, 0, 0])

                for k in range(5):
                    distance[k] = (next_coords[0] - self.user_center[k, 0])**2 + (next_coords[1] - self.user_center[k, 1])**2

                o = distance.tolist().index(distance.min())

                eh = p * (u[o] / np.emath.log2(1 + ((p*(rho / (100**2 + distance.min())))/sigma))) * 5

                # 计算能耗
                ec = 0.3 * u[o] / 10

                # 效用函数

                utility = 1 - np.exp(-((u[o] ** 2) / (u[o] + 10)))

                self.battery -= (ef + ec + eh)
                reward = utility - ef - ec - eh

                if self.battery > 0:
                    done = False
                elif next_coords == [400, 400]:
                    done = True

                s_ = np.hstack((np.array([next_coords[0], next_coords[1],
                                 self.battery])))
                return s_, reward, done, o

    def render(self):
        # time.sleep(0.5)
        self.update()


class QTable:
    def __init__(self,
                 actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 ):
        self.actions = actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

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
    def choose_action(self, observation):        # 由m产生随机数
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:           # 选最优动作则flag=1
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:            # 随机选动作则flag=0
            action = np.random.choice(self.actions)

        return action

    # 学习过程

    def learn(self, s, a, r, s_):      # alpha 为学习率，g为平均奖励率，tau为step计数器

        self.check_state_exist(s_)

        q_predict = self.q_table.ix[s, a]

        q_target = r + self.gamma * self.q_table.ix[s_, :].max() - self.q_table.ix[s, a]

        self.q_table.ix[s, a] = (1 - self.lr) * q_predict + self.lr * q_target

        return self.q_table


start = time.time()

# 回合次数
N = 10


# 主程序
def run_maze():

    # ---------------------------------Q-Learning+人群固定-----------------------------------------------
    remark0 = []
    average_remark0 = []
    cumulative_reward0 = 0

    migration0 = []
    average_migration0 = []
    cumulative_migration0 = 0

    t0 = 0  # updates times

    global q_table

    for episode in range(N):

        s0 = []
        a0 = []
        user = []

        observation = env.reset_uav()

        while True:

            s0.append(observation)

            env.render()

            action = RL1.choose_action(str(observation))

            a0.append(action)

            observation_, reward, done, u = env.step(action)  # tau为step计数器

            user.append(u)

            q_table = RL1.learn(str(observation), action, reward, str(observation_))

            t0 += 1  # 更新次数加一

            cumulative_reward0 += reward  # 累计奖励

            if done:
                print("episode, state", episode, s0)
                print("episode, action", episode, user)

                remark0.append(cumulative_reward0)                    # 记录累计奖励
                average_remark0.append(cumulative_reward0 / t0)       # 记录平均奖励率

                break

            observation = observation_

    # --------------------------------------平均吞吐量对比图-------------------------------------
    plt.figure(figsize=(10, 8))
    # plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    x0 = []
    g0 = []

    for i in range(len(average_migration0)):
        if i % 50 == 0:
            x0.append(np.arange(len(average_migration0))[i])
            g0.append(average_migration0[i])

    plt.plot(x0, g0, marker='o', label='Q_Learning ', markersize=10)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    plt.legend(loc=4)

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}

    plt.ylabel('Average Throughput', font2)
    plt.xlabel('Training Episodes', font2)
    plt.savefig('average_migration.eps', bbox_inches='tight')
    plt.show()

    env.render()
    end = time.time()
    print("game over!")
    print('运行时间:', end - start)
    engine = pyttsx3.init()
    engine.say('程序运行完成')
    engine.runAndWait()
    # env.destory()


if __name__ == "__main__":
    env = Maze()
    RL1 = QTable(actions=list(range(env.n_actions)))

    env.after(100, run_maze)
    env.mainloop()