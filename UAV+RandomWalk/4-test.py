# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/7/31

# Q_learning + 高斯马尔科夫移动模型 + 自己调整，如何控制QoS + 用户关联(a1,b1)
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

    global USER

    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
        #self.action_space = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-12', '1-13', '1-14', '1-15', '1-16', '1-17', '1-18', '1-19', '1-20', '1-21', '1-22', '1-23', '1-24', '1-25',
        #                     '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12', '2-13', '2-14', '2-15', '2-16', '2-17', '2-18', '2-19', '2-20', '2-21', '2-22', '2-23', '2-24', '2-25',
        #                     '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11', '3-12', '3-13', '3-14', '3-15', '3-16', '3-17', '3-18', '3-19', '3-20', '3-21', '3-22', '3-23', '3-24', '3-25',
        #                     '4-1', '4-2', '4-3', '4-4', '4-5', '4-6', '4-7', '4-8', '4-9', '4-10', '4-11', '4-12', '4-13', '4-14', '4-15', '4-16', '4-17', '4-18', '4-19', '4-20', '4-21', '4-22', '4-23', '4-24', '4-25',
        #                     '5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7', '5-8', '5-9', '5-10', '5-11', '5-12', '5-13', '5-14', '5-15', '5-16', '5-17', '5-18', '5-19', '5-20', '5-21', '5-22', '5-23', '5-24', '5-25',
        #                     '6-1', '6-2', '6-3', '6-4', '6-5', '6-6', '6-7', '6-8', '6-9', '6-10', '6-11', '6-12', '6-13', '6-14', '6-15', '6-16', '6-17', '6-18', '6-19', '6-20', '6-21', '6-22', '6-23', '6-24', '6-25',
        #                     '7-1', '7-2', '7-3', '7-4', '7-5', '7-6', '7-7', '7-8', '7-9', '7-10', '7-11', '7-12', '7-13', '7-14', '7-15', '7-16', '7-17', '7-18', '7-19', '7-20', '7-21', '7-22', '7-23', '7-24', '7-25',
        #                     '8-1', '8-2', '8-3', '8-4', '8-5', '8-6', '8-7', '8-8', '8-9', '8-10', '8-11', '8-12', '8-13', '8-14', '8-15', '8-16', '8-17', '8-18', '8-19', '8-20', '8-21', '8-22', '8-23', '8-24', '8-25',
        #                     '9-1', '9-2', '9-3', '9-4', '9-5', '9-6', '9-7', '9-8', '9-9', '9-10', '9-11', '9-12', '9-13', '9-14', '9-15', '9-16', '9-17', '9-18', '9-19', '9-20', '9-21', '9-22', '9-23', '9-24', '9-25',
        #                     '10-1', '10-2', '10-3', '10-4', '10-5', '10-6', '10-7', '10-8', '10-9', '10-10', '10-11', '10-12', '10-13', '10-14', '10-15', '10-16', '10-17', '10-18', '10-19', '10-20', '10-21', '10-22', '10-23', '10-24', '10-25',
        #                     '11-1', '11-2', '11-3', '11-4', '11-5', '11-6', '11-7', '11-8', '11-9', '11-10', '11-11', '11-12', '11-13', '11-14', '11-15', '11-16', '11-17', '11-18', '11-19', '11-20', '11-21', '11-22', '11-23', '11-24', '11-25',
        #                     '12-1', '12-2', '12-3', '12-4', '12-5', '12-6', '12-7', '12-8', '12-9', '12-10', '12-11', '12-12', '12-13', '12-14', '12-15', '12-16', '12-17', '12-18', '12-19', '12-20', '12-21', '12-22', '12-23', '12-24', '12-25',
        #                     '13-1', '13-2', '13-3', '13-4', '13-5', '13-6', '13-7', '13-8', '13-9', '13-10', '13-11', '13-12', '13-13', '13-14', '13-15', '13-16', '13-17', '13-18', '13-19', '13-20', '13-21', '13-22', '13-23', '13-24', '13-25',
        #                     '14-1', '14-2', '14-3', '14-4', '14-5', '14-6', '14-7', '14-8', '14-9', '14-10', '14-11', '14-12', '14-13', '14-14', '14-15', '14-16', '14-17', '14-18', '14-19', '14-20', '14-21', '14-22', '14-23', '14-24', '14-25',
        #                     '15-1', '15-2', '15-3', '15-4', '15-5', '15-6', '15-7', '15-8', '15-9', '15-10', '15-11', '15-12', '15-13', '15-14', '15-15', '15-16', '15-17', '15-18', '15-19', '15-20', '15-21', '15-22', '15-23', '15-24', '15-25',
        #                     ]
        self.n_actions = len(self.action_space)      # 25个动作
        self.n_features = 3                          # 无人机位置, 电池量，信道
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
        for i in range(15):
            self.canvas.create_oval(self.user_center[i, 0] - 5, self.user_center[i, 1] - 5,
                                    self.user_center[i, 0] + 5, self.user_center[i, 1] + 5,
                                    fill='black')

        # --------------------------UAV标识-----------------------------
        self.img = tk.PhotoImage(file="UAV.png")
        self.uav = self.canvas.create_image((40, 40), image=self.img)

        # pack all
        self.canvas.pack()

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

    def reset_uav(self):
        self.update()
        time.sleep(0.1)
        self.battery = 10
        USER = []
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
        for j in range(15):
            self.canvas.create_oval(self.user_center[j, 0] - 5, self.user_center[j, 1] - 5,
                                    self.user_center[j, 0] + 5, self.user_center[j, 1] + 5,
                                    fill='black')

        self.uav = self.canvas.create_image((40, 40), image=self.img)

        return np.hstack((np.array([self.canvas.coords(self.uav)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.uav)[1] / (MAZE_H * UNIT)]), self.battery / 10))

    def step(self, action):
        global USER
        USER = []
        s = np.array(self.canvas.coords(self.uav))               # 无人机位置
        for i in range(25):
            if action == i:
                self.canvas.delete(self.uav)
                point = self.oval_center[i, :]
                self.uav = self.canvas.create_image((point[0], point[1]), image=self.img)
                break

        next_coords = np.array(self.canvas.coords(self.uav), self.user_center[:])           # 下一个状态,返回值是列表[ , ]

        u = np.random.uniform(0, 10, 15)
        # u = [2, 4, 6, 5, 7, 9, 10, 8, 6, 5, 3, 9, 10, 5, 7]

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

                USER.append(o)

                eh = p * (u[o] / np.emath.log2(1 + ((p*(rho / (50**2 + distance.min())))/sigma))) * 5

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
N = 100


# 主程序
def update():
    # main part of RL loop

    remark0 = []
    average_remark0 = []
    migration0 = []
    average_migration0 = []

    s = []
    a = []
    USER = []

    cumulative_reward0 = 0
    cumulative_migration0 = 0
    t = 0  # updates time

    global q_table

    for episode in range(N):

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

            observation_, reward, done, u = env.step(action)  # tau为step计数器

            q_table = RL.learn(str(observation), action, reward, str(observation_))

            t += 1  # 更新次数加一
            T += 1

            cumulative_reward0 += reward  # 累计奖励
            cumulative_migration0 += u    # 累计吞吐量

            if done:
                remark0.append(cumulative_reward0)                    # 记录累计奖励
                average_remark0.append(cumulative_reward0 / t)        # 记录平均奖励率

                migration0.append(cumulative_migration0)              # 记录累计吞吐量
                average_migration0.append(cumulative_migration0 / t)  # 记录平均吞吐量
                break

            observation = observation_

    # --------------------------------------平均吞吐量对比图-------------------------------------
    plt.figure(figsize=(10, 8))
    # plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    x0 = []
    g0 = []

    for i in range(len(average_migration0)):
        if i % 5 == 0:
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
    # --------------------------------------平均reward对比图-------------------------------------
    plt.figure(figsize=(10, 8))
    # plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    x0 = []
    g0 = []

    for i in range(len(average_migration0)):
        if i % 5 == 0:
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
    RL = QTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
