# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/25

# Q_learning + 高斯马尔科夫移动模型 + 用户关联
# 10000次 内存不足

import pandas as pd

import tkinter as tk

import time

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns


np.random.seed(2)               # reproducible

V = [10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20]                # 人群移动速度
W = [30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 330, 360]     # 人群移动角度

UNIT = 40
MAZE_H = 15
MAZE_W = 25

N_STATES = 25                   # the length of the 1 dimensional world

ACTIONS = ['01-01', '01-02', '01-03', '01-04', '01-05', '01-06', '01-07', '01-08', '01-09', '01-10', '01-11', '01-12', '01-13', '01-14', '01-15', '01-16', '01-17', '01-18', '01-19', '01-20', '01-21', '01-22', '01-23', '01-24', '01-25',
           '02-01', '02-02', '02-03', '02-04', '02-05', '02-06', '02-07', '02-08', '02-09', '02-10', '02-11', '02-12', '02-13', '02-14', '02-15', '02-16', '02-17', '02-18', '02-19', '02-20', '02-21', '02-22', '02-23', '02-24', '02-25',
           '03-01', '03-02', '03-03', '03-04', '03-05', '03-06', '03-07', '03-08', '03-09', '03-10', '03-11', '03-12', '03-13', '03-14', '03-15', '03-16', '03-17', '03-18', '03-19', '03-20', '03-21', '03-22', '03-23', '03-24', '03-25',
           '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '04-07', '04-08', '04-09', '04-10', '04-11', '04-12', '04-13', '04-14', '04-15', '04-16', '04-17', '04-18', '04-19', '04-20', '04-21', '04-22', '04-23', '04-24', '04-25',
           '05-01', '05-02', '05-03', '05-04', '05-05', '05-06', '05-07', '05-08', '05-09', '05-10', '05-11', '05-12', '05-13', '05-14', '05-15', '05-16', '05-17', '05-18', '05-19', '05-20', '05-21', '05-22', '05-23', '05-24', '05-25',
           '06-01', '06-02', '06-03', '06-04', '06-05', '06-06', '06-07', '06-08', '06-09', '06-10', '06-11', '06-12', '06-13', '06-14', '06-15', '06-16', '06-17', '06-18', '06-19', '06-20', '06-21', '06-22', '06-23', '06-24', '06-25',
           '07-01', '07-02', '07-03', '07-04', '07-05', '07-06', '07-07', '07-08', '07-09', '07-10', '07-11', '07-12', '07-13', '07-14', '07-15', '07-16', '07-17', '07-18', '07-19', '07-20', '07-21', '07-22', '07-23', '07-24', '07-25',
           '08-01', '08-02', '08-03', '08-04', '08-05', '08-06', '08-07', '08-08', '08-09', '08-10', '08-11', '08-12', '08-13', '08-14', '08-15', '08-16', '08-17', '08-18', '08-19', '08-20', '08-21', '08-22', '08-23', '08-24', '08-25',
           '09-01', '09-02', '09-03', '09-04', '09-05', '09-06', '09-07', '09-08', '09-09', '09-10', '09-11', '09-12', '09-13', '09-14', '09-15', '09-16', '09-17', '09-18', '09-19', '09-20', '09-21', '09-22', '09-23', '09-24', '09-25',
           '10-01', '10-02', '10-03', '10-04', '10-05', '10-06', '10-07', '10-08', '10-09', '10-10', '10-11', '10-12', '10-13', '10-14', '10-15', '10-16', '10-17', '10-18', '10-19', '10-20', '10-21', '10-22', '10-23', '10-24', '10-25',
           '11-01', '11-02', '11-03', '11-04', '11-05', '11-06', '11-07', '11-08', '11-09', '11-10', '11-11', '11-12', '11-13', '11-14', '11-15', '11-16', '11-17', '11-18', '11-19', '11-20', '11-21', '11-22', '11-23', '11-24', '11-25',
           '12-01', '12-02', '12-03', '12-04', '12-05', '12-06', '12-07', '12-08', '12-09', '12-10', '12-11', '12-12', '12-13', '12-14', '12-15', '12-16', '12-17', '12-18', '12-19', '12-20', '12-21', '12-22', '12-23', '12-24', '12-25',
           '13-01', '13-02', '13-03', '13-04', '13-05', '13-06', '13-07', '13-08', '13-09', '13-10', '13-11', '13-12', '13-13', '13-14', '13-15', '13-16', '13-17', '13-18', '13-19', '13-20', '13-21', '13-22', '13-23', '13-24', '13-25',
           '14-01', '14-02', '14-03', '14-04', '14-05', '14-06', '14-07', '14-08', '14-09', '14-10', '14-11', '14-12', '14-13', '14-14', '14-15', '14-16', '14-17', '14-18', '14-19', '14-20', '14-21', '14-22', '14-23', '14-24', '14-25',
           '15-01', '15-02', '15-03', '15-04', '15-05', '15-06', '15-07', '15-08', '15-09', '15-10', '15-11', '15-12', '15-13', '15-14', '15-15', '15-16', '15-17', '15-18', '15-19', '15-20', '15-21', '15-22', '15-23', '15-24', '15-25']

N_actions = len(ACTIONS)        # available actions

EPSILON = 0.9                   # greedy police

ALPHA = 0.1                     # learning rate

GAMMA = 0.9                     # discount factor

MAX_EPISODES = 100              # maximum episodes

FRESH_TIME = 0.3                # fresh time for one move


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['01-01', '01-02', '01-03', '01-04', '01-05', '01-06', '01-07', '01-08', '01-09', '01-10',
                             '01-11', '01-12', '01-13', '01-14', '01-15', '01-16', '01-17', '01-18', '01-19', '01-20',
                             '01-21', '01-22', '01-23', '01-24', '01-25',
                             '02-01', '02-02', '02-03', '02-04', '02-05', '02-06', '02-07', '02-08', '02-09', '02-10',
                             '02-11', '02-12', '02-13', '02-14', '02-15', '02-16', '02-17', '02-18', '02-19', '02-20',
                             '02-21', '02-22', '02-23', '02-24', '02-25',
                             '03-01', '03-02', '03-03', '03-04', '03-05', '03-06', '03-07', '03-08', '03-09', '03-10',
                             '03-11', '03-12', '03-13', '03-14', '03-15', '03-16', '03-17', '03-18', '03-19', '03-20',
                             '03-21', '03-22', '03-23', '03-24', '03-25',
                             '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '04-07', '04-08', '04-09', '04-10',
                             '04-11', '04-12', '04-13', '04-14', '04-15', '04-16', '04-17', '04-18', '04-19', '04-20',
                             '04-21', '04-22', '04-23', '04-24', '04-25',
                             '05-01', '05-02', '05-03', '05-04', '05-05', '05-06', '05-07', '05-08', '05-09', '05-10',
                             '05-11', '05-12', '05-13', '05-14', '05-15', '05-16', '05-17', '05-18', '05-19', '05-20',
                             '05-21', '05-22', '05-23', '05-24', '05-25',
                             '06-01', '06-02', '06-03', '06-04', '06-05', '06-06', '06-07', '06-08', '06-09', '06-10',
                             '06-11', '06-12', '06-13', '06-14', '06-15', '06-16', '06-17', '06-18', '06-19', '06-20',
                             '06-21', '06-22', '06-23', '06-24', '06-25',
                             '07-01', '07-02', '07-03', '07-04', '07-05', '07-06', '07-07', '07-08', '07-09', '07-10',
                             '07-11', '07-12', '07-13', '07-14', '07-15', '07-16', '07-17', '07-18', '07-19', '07-20',
                             '07-21', '07-22', '07-23', '07-24', '07-25',
                             '08-01', '08-02', '08-03', '08-04', '08-05', '08-06', '08-07', '08-08', '08-09', '08-10',
                             '08-11', '08-12', '08-13', '08-14', '08-15', '08-16', '08-17', '08-18', '08-19', '08-20',
                             '08-21', '08-22', '08-23', '08-24', '08-25',
                             '09-01', '09-02', '09-03', '09-04', '09-05', '09-06', '09-07', '09-08', '09-09', '09-10',
                             '09-11', '09-12', '09-13', '09-14', '09-15', '09-16', '09-17', '09-18', '09-19', '09-20',
                             '09-21', '09-22', '09-23', '09-24', '09-25',
                             '10-01', '10-02', '10-03', '10-04', '10-05', '10-06', '10-07', '10-08', '10-09', '10-10',
                             '10-11', '10-12', '10-13', '10-14', '10-15', '10-16', '10-17', '10-18', '10-19', '10-20',
                             '10-21', '10-22', '10-23', '10-24', '10-25',
                             '11-01', '11-02', '11-03', '11-04', '11-05', '11-06', '11-07', '11-08', '11-09', '11-10',
                             '11-11', '11-12', '11-13', '11-14', '11-15', '11-16', '11-17', '11-18', '11-19', '11-20',
                             '11-21', '11-22', '11-23', '11-24', '11-25',
                             '12-01', '12-02', '12-03', '12-04', '12-05', '12-06', '12-07', '12-08', '12-09', '12-10',
                             '12-11', '12-12', '12-13', '12-14', '12-15', '12-16', '12-17', '12-18', '12-19', '12-20',
                             '12-21', '12-22', '12-23', '12-24', '12-25',
                             '13-01', '13-02', '13-03', '13-04', '13-05', '13-06', '13-07', '13-08', '13-09', '13-10',
                             '13-11', '13-12', '13-13', '13-14', '13-15', '13-16', '13-17', '13-18', '13-19', '13-20',
                             '13-21', '13-22', '13-23', '13-24', '13-25',
                             '14-01', '14-02', '14-03', '14-04', '14-05', '14-06', '14-07', '14-08', '14-09', '14-10',
                             '14-11', '14-12', '14-13', '14-14', '14-15', '14-16', '14-17', '14-18', '14-19', '14-20',
                             '14-21', '14-22', '14-23', '14-24', '14-25',
                             '15-01', '15-02', '15-03', '15-04', '15-05', '15-06', '15-07', '15-08', '15-09', '15-10',
                             '15-11', '15-12', '15-13', '15-14', '15-15', '15-16', '15-17', '15-18', '15-19', '15-20',
                             '15-21', '15-22', '15-23', '15-24', '15-25']

        self.n_actions = len(self.action_space)
        self.battery = 10
        self.title('UAV测试环境-Q-GMR-用户关联')
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
        self.uav_center = np.array([[100, 100], [220, 60], [500, 100], [220, 180], [380, 220],
                                     [180, 300], [60, 380], [220, 380], [460, 380], [100, 460],
                                     [340, 460], [220, 540], [500, 540], [660, 20], [780, 60],
                                     [980, 220], [700, 140], [580, 180], [900, 220], [620, 260],
                                     [940, 340], [820, 380], [660, 460], [940, 540], [700, 580]])
        for i in range(25):
            self.canvas.create_oval(self.uav_center[i, 0] - 10, self.uav_center[i, 1] - 10,
                                    self.uav_center[i, 0] + 10, self.uav_center[i, 1] + 10,
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

    def reset_uav(self):
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
        self.uav_center = np.array([[100, 100], [220, 60], [500, 100], [220, 180], [380, 220],
                                     [180, 300], [60, 380], [220, 380], [460, 380], [100, 460],
                                     [340, 460], [220, 540], [500, 540], [660, 20], [780, 60],
                                     [980, 220], [700, 140], [580, 180], [900, 220], [620, 260],
                                     [940, 340], [820, 380], [660, 460], [940, 540], [700, 580]])
        for i in range(25):
            self.canvas.create_oval(self.uav_center[i, 0] - 10, self.uav_center[i, 1] - 10,
                                    self.uav_center[i, 0] + 10, self.uav_center[i, 1] + 10,
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

        return np.hstack((np.array([[self.canvas.coords(self.uav)[0], self.canvas.coords(self.uav)[1]]]).reshape(1, 2),
                          np.array([self.user_center]).reshape(1, 30)))

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

        return self.user_center[:], speed[:], angle[:]

    def step(self, observation, action, t):

        global V
        global W

        s = observation                                                          # 当前状态：无人机位置以及15个用户的位置, 16*2, [[....]]

        a = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']

        o = 0                                                                    # 用户位置索引初始化
        next_coords = []                                                         # 无人机下一个位置初始化

        print("用户：", self.action_space[action][0:2])
        print("无人机：", self.action_space[action][3:5])

        if self.action_space[action][0:2] in a:                                  # 找到对应动作的用户序号索引值
            o = a.index(self.action_space[action][0:2])                          # 选择服务的用户的索引

        if self.action_space[action][3:5] in a:                                  # 找到对应动作的无人机位置索引值，更新无人机的位置
            self.canvas.delete(self.uav)
            i = a.index(self.action_space[action][3:5])
            next_coords.append(list(self.uav_center[i, :]))                      # array 类型数据, [[....]]
            self.uav = self.canvas.create_image((next_coords[0][0], next_coords[0][1]), image=self.img)

        u = np.random.uniform(0, 10, 15)                                         # 随机到达用户任务量

        # -----------------------------计算奖励函数-归一化问题----------------------------------
        p = 0.1
        rho = 10e-5
        sigma = 9.999999999999987e-15

        # 飞行能耗
        ef = p * (np.sqrt((s[0][0] - next_coords[0][0]) ** 2 + (s[0][1] - next_coords[0][1]) ** 2)) / 800

        # 盘旋能耗
        distance = (next_coords[0][0] - self.user_center[o, 0])**2 + (next_coords[0][1] - self.user_center[o, 1])**2

        eh = p * (u[o] / np.emath.log2(1 + ((p*(rho / (100**2 + distance)))/sigma))) * 5

        # 计算能耗
        ec = 0.3 * u[o] / 10

        # 效用函数
        utility = 1 - np.exp(-((u[o] ** 2) / (u[o] + 10)))

        # 剩余电池量
        self.battery -= (ef + ec + eh)

        # 奖励
        reward = utility - ef - ec - eh

        # -------------------------------------本回合是否结束---------------------------------
        if self.battery <= p*(np.sqrt((40-next_coords[0][0])**2+(40-next_coords[0][1])**2))/800:
            done = True
        else:
            done = False

        # --------------------------------------下一个状态------------------------------------------
        self.user_center[:], V0, W0 = self.update_env(V, W, t)                  # 更新用户位置
        V = V0
        W = W0
        print("速度：", V)
        print("角度：", W)

        s_ = np.hstack((np.array([[next_coords[0][0], next_coords[0][1]]]).reshape(1, 2), np.array([self.user_center]).reshape(1, 30)))  # 1*32的数据， [[....]]

        return s_, reward, done, u[o]

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

    cumulative_reward = 0
    cumulative_migration = 0
    t = 0  # updates time

    global q_table

    for episode in range(2000):

        T = 0

        observation = env.reset_uav()

        while True:

            s.append(observation)

            env.render()

            action = RL.choose_action(str(observation))

            a.append(action)

            observation_, reward, done, u = env.step(observation, action, T)  # tau为step计数器

            q_table = RL.learn(str(observation), action, reward, str(observation_))

            t += 1  # 更新次数加一
            T += 1

            cumulative_migration += u  # 累计吞吐量

            cumulative_reward += reward  # 累计奖励

            if done:
                print("------------------------------------回合数:", episode)
                print("====================================Q_Table尺寸", np.shape(q_table))
                remark.append(cumulative_reward)                    # 记录累计奖励
                average_remark.append(cumulative_reward / t)        # 记录平均奖励率
                migration.append(cumulative_migration)              # 记录累计吞吐量
                average_migration.append(cumulative_migration / t)  # 记录平均吞吐量
                break

            observation = observation_

    # 平均奖励
    plt.figure(figsize=(10, 8))

    x1 = []
    g1 = []
    for i in range(len(average_remark)):
        if i % 10 == 0:
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
        if i % 10 == 0:
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