# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/7/28

# Q_learning + 固定人群

import numpy as np
import tkinter as tk
import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pyttsx3

UNIT = 40
MAZE_H = 15
MAZE_W = 25

np.random.seed(1)
tf.set_random_seed(1)


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                             '11', '12', '13', '14', '15', '16', '17', '18', '19',
                             '20', '21', '22', '23', '24', '25']
        self.n_actions = len(self.action_space)
        self.n_features = 3  # 无人机位置, 电池量，信道
        self.title('UAV测试环境-Q')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # --------------------------构建固定点---------------------------
        self.oval_center = np.array([[100, 100], [220, 60], [500, 100], [220, 180], [380, 220],
                                     [180, 300], [60, 380], [220, 380], [460, 380], [100, 460],
                                     [340, 460], [220, 540], [500, 540], [660, 20], [780, 60],
                                     [980, 220], [700, 140], [580, 180], [900, 220], [620, 260],
                                     [940, 340], [820, 380], [660, 460], [940, 540], [700, 580]])
        for i in range(25):
            self.canvas.create_oval(self.oval_center[i, 0] - 10, self.oval_center[i, 1] - 10,
                                    self.oval_center[i, 0] + 10, self.oval_center[i, 1] + 10,
                                    fill='blue')

        # --------------------------用户--------------------------------
        self.user_center = np.array([[380, 100], [180, 140], [100, 260], [500, 260], [340, 340],
                                     [140, 380], [60, 540], [340, 580], [940, 20], [620, 100],
                                     [860, 140], [740, 300], [580, 420], [980, 420], [780, 500]])
        for i in range(len(self.user_center)):
            self.canvas.create_oval(self.user_center[i, 0] - 5, self.user_center[i, 1] - 5,
                                    self.user_center[i, 0] + 5, self.user_center[i, 1] + 5,
                                    fill='black')

        # --------------------------UAV标识-----------------------------
        self.img = tk.PhotoImage(file="UAV.png")
        self.uav = self.canvas.create_image((40, 40), image=self.img)

        # pack all
        self.canvas.pack()

    """
    def update_env(self):
        for i in range(len(self.user_center)):
            if np.random.uniform() <= 0.25:                                    # 向前
                if self.user_center[i, 0] >= 960:
                    self.user_center[i, 0] = self.user_center[i, 0] - 40
                else:
                    self.user_center[i, 0] = self.user_center[i, 0] + 40

            elif 0.25 < np.random.uniform() <= 0.5:                            # 向后
                if self.user_center[i, 0] <= 40:
                    self.user_center[i, 0] = self.user_center[i, 0] + 40
                else:
                    self.user_center[i, 0] = self.user_center[i, 0] - 40

            elif 0.5 < np.random.uniform() <= 0.75:                            # 向上
                if self.user_center[i, 1] >= 560:
                    self.user_center[i, 1] = self.user_center[i, 0] - 40
                else:
                    self.user_center[i, 1] = self.user_center[i, 1] + 40

            elif np.random.uniform() <= 1:                                     # 向下
                if self.user_center[i, 1] <= 40:
                    self.user_center[i, 1] = self.user_center[i, 0] + 40
                else:
                    self.user_center[i, 1] = self.user_center[i, 1] - 40

        for i in range(len(self.user_center)):
            self.canvas.create_oval(max(0, self.user_center[i, 0] - 5), max(self.user_center[i, 1] - 5, 0),
                                        min(1000, self.user_center[i, 0] + 5), min(self.user_center[i, 1] + 5, 600),
                                        fill='red')
    """

    def reset_uav(self):
        self.update()
        time.sleep(0.1)
        self.battery = 10
        self.canvas.delete(self.uav)
        self.canvas.create_rectangle(0, 0, 1000, 600, fill='white')

        # ------------------------------create grids，画表格-------------------------------

        for c in range(0, MAZE_W * UNIT, 5 * UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W * UNIT

            self.canvas.create_line(x0, y0, x1, y1)

        for r in range(0, MAZE_H * UNIT, 5 * UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r

            self.canvas.create_line(x0, y0, x1, y1)

        # --------------------------构建固定点-------------------------------------------------
        self.oval_center = np.array([[100, 100], [220, 60], [500, 100], [220, 180], [380, 220],
                                     [180, 300], [60, 380], [220, 380], [460, 380], [100, 460],
                                     [340, 460], [220, 540], [500, 540], [660, 20], [780, 60],
                                     [980, 220], [700, 140], [580, 180], [900, 220], [620, 260],
                                     [940, 340], [820, 380], [660, 460], [940, 540], [700, 580]])
        for i in range(25):
            self.canvas.create_oval(self.oval_center[i, 0] - 10, self.oval_center[i, 1] - 10,
                                    self.oval_center[i, 0] + 10, self.oval_center[i, 1] + 10,
                                    fill='blue')

        # --------------------------用户-------------------------------------------------------------
        self.user_center = np.array([[380, 100], [180, 140], [100, 260], [500, 260], [340, 340],
                                     [140, 380], [60, 540], [340, 580], [940, 20], [620, 100],
                                     [860, 140], [740, 300], [580, 420], [980, 420], [780, 500]])
        for j in range(len(self.user_center)):
            self.canvas.create_oval(self.user_center[j, 0] - 5, self.user_center[j, 1] - 5,
                                    self.user_center[j, 0] + 5, self.user_center[j, 1] + 5,
                                    fill='black')

        self.uav = self.canvas.create_image((40, 40), image=self.img)

        return np.hstack((np.array([self.canvas.coords(self.uav)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.uav)[1] / (MAZE_H * UNIT)]), self.battery / 10))

    def step(self, action):
        s = np.array(self.canvas.coords(self.uav))
        for i in range(25):
            if action == i:
                self.canvas.delete(self.uav)
                point = self.oval_center[i, :]
                self.uav = self.canvas.create_image((point[0], point[1]), image=self.img)
                break

        next_coords = self.canvas.coords(self.uav)  # next state,返回值是列表[ , ]

        u = np.random.uniform(0, 10, 15)

        # reward function
        for j in range(25):
            if next_coords == self.oval_center[j, :].tolist():
                p = 0.1
                rho = 10e-5
                sigma = 9.999999999999987e-15
                # 飞行能耗，归一化问题
                ef = p * (np.sqrt((s[0] - self.oval_center[j, 0]) ** 2 +
                                  (s[1] - self.oval_center[j, 1]) ** 2)) / 80

                # 盘旋能耗
                distance = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                for k in range(15):
                    distance[k] = (next_coords[0] - self.user_center[k, 0])**2 + (next_coords[1] - self.user_center[k, 1])**2

                o = distance.tolist().index(distance.min())

                self.user_center.remove(o)

                eh = p * (u[o] / np.emath.log2(1 + ((p*(rho / (100**2 + distance.min())))/sigma))) * 5

                # 计算能耗
                ec = 0.3 * u[o] / 10

                # 效用函数

                utility = 1 - np.exp(-((u[o] ** 2) / (u[o] + 10)))

                self.battery -= (ef + ec + eh)
                reward = utility - ef - ec - eh

                if self.battery <= p*(np.sqrt((40-self.oval_center[j, 0])**2+(40-self.oval_center[j, 1])**2))/80:
                    done = True
                else:
                    done = False
                # s_ = np.array([next_coords[0] / (MAZE_H * UNIT), next_coords[1] / (MAZE_W * UNIT)])
                s_ = np.hstack((np.array([next_coords[0] / (MAZE_H * UNIT), next_coords[1] / (MAZE_W * UNIT)]),
                                 self.battery / 10))
                return s_, reward, done, u[o]

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
N = 1000


# 主程序
def run_maze():

    # ---------------------------------Q-Learning+人群固定-----------------------------------------------
    remark0 = []
    average_remark0 = []
    cumulative_reward0 = 0

    migration0 = []
    average_migration0 = []
    cumulative_migration0 = 0

    s0 = []
    a0 = []

    t0 = 0  # updates times

    global q_table

    for episode in range(N):

        observation = env.reset_uav()

        while True:

            s0.append(observation)

            env.render()

            action = RL1.choose_action(str(observation))

            a0.append(action)

            observation_, reward, done, u = env.step(action)  # tau为step计数器

            q_table = RL1.learn(str(observation), action, reward, str(observation_))

            t0 += 1  # 更新次数加一

            cumulative_reward0 += reward  # 累计奖励
            cumulative_migration0 += u    # 累计吞吐量

            if done:
                remark0.append(cumulative_reward0)                    # 记录累计奖励
                average_remark0.append(cumulative_reward0 / t0)        # 记录平均奖励率

                migration0.append(cumulative_migration0)              # 记录累计吞吐量
                average_migration0.append(cumulative_migration0 / t0)  # 记录平均吞吐量
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
