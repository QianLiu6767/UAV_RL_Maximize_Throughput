# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/3

# Q_learning + 高斯马尔科夫移动模型
# 100000次不收敛

import pandas as pd

import tkinter as tk

import time

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns


np.random.seed(2)               # reproducible

UNIT = 40
MAZE_H = 15
MAZE_W = 25

N_STATES = 25                    # the length of the 1 dimensional world

ACTIONS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']

N_actions = len(ACTIONS)     # available actions

EPSILON = 0.9                   # greedy police

ALPHA = 0.1                     # learning rate

GAMMA = 0.9                     # discount factor

MAX_EPISODES = 100               # maximum episodes

FRESH_TIME = 0.3                # fresh time for one move


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                             '11', '12', '13', '14', '15', '16', '17', '18', '19',
                             '20', '21', '22', '23', '24', '25']
        self.n_actions = len(self.action_space)
        self.battery = 10
        self.title('UAV测试环境-Q_Learning-GMR')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.tau = 0

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # ------------------------------create grids，画表格-------------------------------

        for c1 in range(-20, MAZE_W * UNIT, 5*UNIT):
            x0, y0, x1, y1 = c1, 0, c1, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        for c2 in range(20, MAZE_W * UNIT, 5 * UNIT):
            x0, y0, x1, y1 = c2, 0, c2, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        for r1 in range(-20, MAZE_H * UNIT, 5*UNIT):
            x0, y0, x1, y1 = 0, r1, MAZE_W * UNIT, r1
            self.canvas.create_line(x0, y0, x1, y1)

        for r2 in range(20, MAZE_H * UNIT, 5 * UNIT):
            x0, y0, x1, y1 = 0, r2, MAZE_W * UNIT, r2
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

        # --------------------------用户-------------------------------------------------------
        self.user_center = np.array([[380, 100], [180, 140], [100, 260], [500, 260], [340, 340],
                                     [140, 380], [60, 540], [340, 580], [940, 20], [620, 100],
                                     [860, 140], [740, 300], [580, 420], [980, 420], [780, 500]])
        for j in range(15):
            self.canvas.create_oval(self.user_center[j, 0] - 5, self.user_center[j, 1] - 5,
                                    self.user_center[j, 0] + 5, self.user_center[j, 1] + 5,
                                    fill='black')

        # --------------------------UAV标识-----------------------------------------------------
        self.img = tk.PhotoImage(file="UAV.png")
        self.uav = self.canvas.create_image((40, 40), image=self.img)

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.battery = 10
        self.canvas.delete(self.uav)

        self.canvas.create_rectangle(0, 0, 1000, 600, fill='white')

        # ------------------------------create grids，画表格-------------------------------

        for c1 in range(-20, MAZE_W * UNIT, 5*UNIT):
            x0, y0, x1, y1 = c1, 0, c1, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        for c2 in range(20, MAZE_W * UNIT, 5 * UNIT):
            x0, y0, x1, y1 = c2, 0, c2, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        for r1 in range(-20, MAZE_H * UNIT, 5*UNIT):
            x0, y0, x1, y1 = 0, r1, MAZE_W * UNIT, r1
            self.canvas.create_line(x0, y0, x1, y1)

        for r2 in range(20, MAZE_H * UNIT, 5 * UNIT):
            x0, y0, x1, y1 = 0, r2, MAZE_W * UNIT, r2
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
        for j in range(15):
            self.canvas.create_oval(self.user_center[j, 0] - 5, self.user_center[j, 1] - 5,
                                    self.user_center[j, 0] + 5, self.user_center[j, 1] + 5,
                                    fill='black')

        self.uav = self.canvas.create_image((40, 40), image=self.img)

        return self.canvas.coords(self.uav)

    def update_env(self, speed, angle, t):
        # This is how environment be updated

        for i in range(15):
            speed[i] = 0.9 * speed[i] + (1 - 0.9) * 20 + (1 - 0.9) ** 2 * np.sqrt(speed[i])
            angle[i] = 0.9 * angle[i] + (1 - 0.9) * 180 + (1 - 0.9) ** 2 * np.sqrt(angle[i])

            self.user_center[i, 0] = self.user_center[i, 0] + speed[i] * t * np.cos(angle[i])
            self.user_center[i, 1] = self.user_center[i, 1] + speed[i] * t * np.sin(angle[i])

            if (self.user_center[i, 0] >= 1000) | (self.user_center[i, 1] >= 600):
                self.user_center[i, 0] = 960
                self.user_center[i, 1] = 560
                angle[i] = 360 - angle[i]

        for j in range(15):
            self.canvas.create_oval(max(0, self.user_center[j, 0] - 5), max(self.user_center[j, 1] - 5, 0),
                                        min(1000, self.user_center[j, 0] + 5), min(self.user_center[j, 1] + 5, 600),
                                        fill='red')

        return speed, angle

    def step(self, action):
        s = np.array(self.canvas.coords(self.uav))           # 当前UAV的位置
        for i in range(25):
            if action == i:
                self.canvas.delete(self.uav)                 # 删除当前UAV图片
                point = self.oval_center[i, :]               # 下一个状态的UA v位置
                self.uav = self.canvas.create_image((point[0], point[1]), image=self.img)     # 放置UAV图片
                break

        next_coords = self.canvas.coords(self.uav)  # 下一个状态列表,返回值是列表[ , ]

        u = np.random.uniform(0, 10, 15)

        # reward function
        for j in range(25):
            if next_coords == self.oval_center[j, :].tolist():
                p = 0.1
                rho = 10e-5
                sigma = 9.999999999999987e-15
                # 飞行能耗，归一化问题
                ef = p * (np.sqrt((s[0] - self.oval_center[j, 0]) ** 2 +
                                  (s[1] - self.oval_center[j, 1]) ** 2)) / 800

                # 盘旋能耗
                distance = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                for k in range(15):                               # 计算所有用户到UAV的距离
                    distance[k] = (next_coords[0] - self.user_center[k, 0])**2 + \
                                  (next_coords[1] - self.user_center[k, 1])**2

                o = distance.tolist().index(distance.min())       # 寻找距离UAV最近的用户

                eh = p * (u[o] / np.emath.log2(1 + (p / (((100 ** 2 + distance.min())**2) * sigma))))
                # eh = p * (u[o] / np.emath.log2(1 + ((p*(rho / (100**2 + distance.min())))/sigma))) * 5

                # 计算能耗
                ec = 0.3 * u[o] / 10

                # 效用函数

                utility = 1 - np.exp(-((u[o] ** 2) / (u[o] + 10)))

                self.battery -= (ef + ec + eh)                       # 无人机剩余电池

                reward = utility - ef - ec - eh                      # 当前时隙的奖励

                # 判断本回合是否结束
                if self.battery <= p*(np.sqrt((40-self.oval_center[j, 0])**2+(40-self.oval_center[j, 1])**2))/800:
                    done = True
                else:
                    done = False

                # s_ = np.array([next_coords[0] / (MAZE_H * UNIT), next_coords[1] / (MAZE_W * UNIT)])
                s_ = np.hstack((np.array([next_coords[0], next_coords[1]])))
                return s_, reward, done, u[o], o

    def render(self):
        # time.sleep(0.5)
        self.update()


class Dream:
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


# 主程序
def update():
    # main part of RL loop

    remark = []
    average_remark = []
    migration = []
    average_migration = []

    s = []
    a = []
    USER = []

    cumulative_reward = 0
    cumulative_migration = 0
    t = 0  # updates time

    global q_table

    for episode in range(100000):

        V = [10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20]
        W = [30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 330, 360]
        T = 0

        observation = env.reset()

        while True:

            s.append(observation)

            env.render()

            V, W = env.update_env(V, W, T)

            action = RL.choose_action(str(observation))

            a.append(action)

            observation_, reward, done, u, user = env.step(action)  # tau为step计数器

            USER.append(user)

            q_table = RL.learn(str(observation), action, reward, str(observation_))

            t += 1  # 更新次数加一
            T += 1

            cumulative_migration += u  # 累计吞吐量

            cumulative_reward += reward  # 累计奖励

            if done:
                remark.append(cumulative_reward)  # 记录累计奖励
                average_remark.append(cumulative_reward / t)  # 记录平均奖励率
                migration.append(cumulative_migration)              # 记录累计吞吐量
                average_migration.append(cumulative_migration / t)  # 记录平均吞吐量
                break

            observation = observation_

    # 平均奖励
    plt.figure(figsize=(10, 8))

    x1 = []
    g1 = []
    for i in range(len(average_remark)):
        if i % 1 == 0:
            x1.append(np.arange(len(average_remark))[i])
            g1.append(average_remark[i])

    plt.plot(x1, g1, marker='o', label='BatteryLevel', markersize=10)

    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

    plt.legend(loc=4)

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

    plt.ylabel('Average Return', font2)
    plt.xlabel('Training Episodes', font2)
    plt.savefig('a_r_20000.eps', bbox_inches='tight')
    plt.show()

    # 平均吞吐量
    plt.figure(figsize=(10, 8))
    x2 = []
    g2 = []

    for i in range(len(average_migration)):
        if i % 1 == 0:
            x2.append(np.arange(len(average_migration))[i])
            g2.append(average_migration[i])

    plt.plot(x2, g2, marker='x', label='BatteryLevel2', markersize=10)

    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

    plt.legend(loc=4)

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

    plt.ylabel('Average Throughput', font2)
    plt.xlabel('Training Episodes', font2)
    plt.savefig('a_t_20000.eps', bbox_inches='tight')
    plt.show()

    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = Dream(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()