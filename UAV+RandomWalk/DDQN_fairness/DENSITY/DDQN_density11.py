# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/12/18

# DDQN + 高斯马尔科夫移动模型 + 用户关联（A1, A1）15*25个动作 + 用户速度、角度随机，移动模型中有高斯变量（初始位置变化） + 用户公平性 + 约束无人机最后一点尽可能的近一点
# 改变用户密度-11个用户

import numpy as np
import tkinter as tk
import time
import tensorflow as tf
import math

UNIT = 40
MAZE_H = 15
MAZE_W = 25

V = np.random.uniform(15, 1, 11)  # 人群移动速度
W = np.random.randint(0, 360, 11)  # 人群移动角度


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
                             '11-21', '11-22', '11-23', '11-24', '11-25']
        self.n_actions = len(self.action_space)  # 15*25个动作
        self.n_features = 24  # 无人机位置, 11个用户位置，12*2
        self.title('UAV测试环境-DDQN-11')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # --------------------------构建固定点---------------------------
        self.uav_center = np.array([[100, 100], [220, 60], [500, 100], [220, 180], [380, 220],
                                    [180, 300], [60, 380], [220, 380], [460, 380], [100, 460],
                                    [340, 460], [220, 540], [500, 540], [660, 20], [780, 60],
                                    [980, 220], [700, 140], [580, 180], [900, 220], [620, 260],
                                    [940, 340], [820, 380], [660, 460], [940, 540], [700, 580]])
        for i in range(25):
            self.canvas.create_oval(self.uav_center[i, 0] - 10, self.uav_center[i, 1] - 10,
                                    self.uav_center[i, 0] + 10, self.uav_center[i, 1] + 10,
                                    fill='blue')

        # --------------------------用户--------------------------------
        self.user_center = np.array([[380, 100], [180, 140], [100, 260], [500, 260], [340, 340],
                                     [140, 380], [60, 540], [340, 580], [620, 100],
                                     [740, 300], [580, 420]])
        for i in range(11):
            self.canvas.create_oval(self.user_center[i, 0] - 5, self.user_center[i, 1] - 5,
                                    self.user_center[i, 0] + 5, self.user_center[i, 1] + 5,
                                    fill='black')

        # --------------------------UAV标识-----------------------------
        self.img = tk.PhotoImage(file="UAV.png")
        self.uav = self.canvas.create_image((40, 40), image=self.img)

        # pack all
        self.canvas.pack()

    def update_env(self, speed, angle):

        global V
        global W

        for i in range(11):
            self.user_center[i, 0] = self.user_center[i, 0] + speed[i] * np.cos(angle[i])  # 第ith用户的横坐标
            self.user_center[i, 1] = self.user_center[i, 1] + speed[i] * np.sin(angle[i])  # 第ith用户的纵坐标

            speed[i] = 0.99 * speed[i] + (1 - 0.99) * 18.07 + (1 - 0.99) ** 2 * np.random.normal(1, 0.1, 1)  # 更新ith用户的速度
            angle[i] = 0.95 * angle[i] + (1 - 0.95) * 360 + (1 - 0.95) ** 2 * np.random.normal(180, 50, 1)  # 更新ith用户的角度

            if self.user_center[i, 0] >= 1000:
                self.user_center[i, 0] = 960
                angle[i] = angle[i] + 180
            if self.user_center[i, 0] <= 0:
                self.user_center[i, 0] = 40
                angle[i] = angle[i] + 180

            if self.user_center[i, 1] >= 600:
                self.user_center[i, 1] = 560
                angle[i] = angle[i] + 180
            if self.user_center[i, 1] <= 0:
                self.user_center[i, 1] = 40
                angle[i] = angle[i] + 180

        for j in range(11):
            self.canvas.create_oval(max(0, self.user_center[j, 0] - 5), max(self.user_center[j, 1] - 5, 0),
                                    min(1000, self.user_center[j, 0] + 5), min(self.user_center[j, 1] + 5, 600),
                                    fill='red')
        V = speed
        W = angle
        return self.user_center[:]

    def reset_uav(self):
        self.update()
        time.sleep(0.1)
        self.battery = 200
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
                                     [140, 380], [60, 540], [340, 580], [620, 100],
                                     [740, 300], [580, 420]])
        for j in range(11):
            self.canvas.create_oval(self.user_center[j, 0] - 5, self.user_center[j, 1] - 5,
                                    self.user_center[j, 0] + 5, self.user_center[j, 1] + 5,
                                    fill='black')

        self.uav = self.canvas.create_image((40, 40), image=self.img)

        return np.hstack((np.array([[self.canvas.coords(self.uav)[0], self.canvas.coords(self.uav)[1]]]).reshape(1, 2),
                          np.array([self.user_center]).reshape(1, 22)))

    def step(self, observation, action, FLAG, t, U):

        global V
        global W

        s = observation  # 当前状态：无人机位置以及15个用户的位置, 16*2, [[....]]

        next_coords = []  # 无人机下一个位置初始化

        o = math.ceil(action / 25) - 1  # 用户位置索引,(0,1,2,...,14)
        # print("选择服务的用户：", o+1)

        q = math.ceil(action % 25) - 1  # 找到对应动作的无人机位置索引值，更新无人机的位置
        # print("无人机位置点：", q+1)

        self.canvas.delete(self.uav)  # 删除上一个无人机
        next_coords.append(list(self.uav_center[q, :]))  # array 类型数据, [[....]]
        self.uav = self.canvas.create_image((next_coords[0][0], next_coords[0][1]), image=self.img)

        u = np.random.uniform(0, 10, 15)  # 随机到达用户任务量
        U[o] += u[o]  # 累计用户迁移量

        # -----------------------------计算奖励函数-归一化问题----------------------------------
        p_t = 0.1
        p_f = 0.11

        rho_0 = 9.999999999999987e-6
        sigma = 9.999999999999987e-15
        p_h = 0.08

        C_i = 9.999999999999987e2
        f_c = 2 * 9.999999999999987e8

        H = 50
        N_b = 100

        E_f = 6.4
        E_h = 11
        E_c = 4

        # 飞行能耗
        distance1 = (np.sqrt((s[0, 0] - next_coords[0][0]) ** 2 + (s[0, 1] - next_coords[0][1]) ** 2))
        ef = p_f * distance1 / 20

        # 盘旋能耗
        distance2 = (next_coords[0][0] - self.user_center[o, 0]) ** 2 + (
                    next_coords[0][1] - self.user_center[o, 1]) ** 2
        r_ij = np.emath.log2(1 + ((p_t * (rho_0 / (H ** 2 + distance2))) / sigma))  # 传输速率，7M/s
        eh = p_h * u[o] * N_b / r_ij

        # 计算能耗
        ec = 9.999999999999987e-28 * C_i * f_c ** 2 * u[o] * N_b * 9.999999999999987e2

        # 效用函数
        utility = 1 - np.exp(-((u[o] ** 2) / (u[o] + 10)))

        # 剩余电池量
        self.battery -= (ef + ec + eh)

        # 奖励
        reward = utility - (ef / E_f + ec / E_c + eh / E_h) / 3
        exist = False

        if t > 30:
            for i in range(len(U)):
                if U[i] == 0:
                    exist = True

        if exist:
            Reward = reward * (-100)
        else:
            Reward = reward * 50

        if not FLAG:
            Reward = reward * 100

        # -------------------------------------本回合是否结束---------------------------------
        if self.battery <= p_f * (np.sqrt((40 - next_coords[0][0]) ** 2 + (40 - next_coords[0][1]) ** 2)) / 20:
            done = True
            distance3 = np.sqrt((40 - next_coords[0][0]) ** 2 + (40 - next_coords[0][1]) ** 2)
            if distance3 <= 100:
                Reward = reward + 500
            elif distance3 <= 300:
                Reward = reward + 300
            elif distance3 <= 500:
                Reward = reward + 100
            else:
                Reward = reward
        else:
            done = False

        # --------------------------------------下一个状态------------------------------------------
        self.user_center[:] = self.update_env(V, W)  # 更新用户位置

        s_ = np.hstack((np.array([[next_coords[0][0], next_coords[0][1]]]).reshape(1, 2),
                        np.array([self.user_center]).reshape(1, 22)))  # 1*32的数据， [[....]]

        return s_, reward, Reward, done, u[o]

    def render(self):
        # time.sleep(0.5)
        self.update()


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=50,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数，15*25=375
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  # 每隔300步copy参数
        self.memory_size = memory_size  # 记忆库大小：500
        self.batch_size = batch_size  # 抽样更新大小：50
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0  # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 一共学习了多少步
        self.learn_step_counter = 0

        # 初始化记忆库 [s,a,r,s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # -----------------建立神经网络,两个：evaluate 和 target-------------------------------------------
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        # 是否开启tensorboard， 就是神经网络结构框架图
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ---------------------------- 创建 eval 神经网络, 及时更新参数 ----------------------------------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 用来接收 q_target 的值
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # eval_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # eval_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # -------------------------- 创建 target 神经网络, 提供 target Q -------------------------------------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # target_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # target_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

        # -----------------------------计算神经网络的误差和梯度---------------------------------------------
        with tf.variable_scope('loss'):  # 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):  # 梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------------------保存神经网络的参数---------------------------------------------------
        saver = tf.train.Saver()

    # 储存记忆，从上往下储存，满了就循环
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s.reshape(1, 24), np.array([[a, r]]).reshape(1, 2), s_.reshape(1, 24)))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 选择动作，90%使用e-greedy policy选择，10%使用随机办法
    def choose_action(self, observation, U):

        flag = False

        for i in range(len(U)):
            if U[i] < 5:
                flag = True

        while flag:

            if np.random.uniform() < self.epsilon:  # 375个动作里随机选Q值最大的一个动作
                # forward feed the observation and get q value for every actions,向前输出 这里使用的是估计神经网络
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation[:]})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)  # 375个动作里随机选一个

            o = math.ceil(action / 25) - 1  # 用户位置索引

            if U[o] >= 5:
                flag = True
            else:
                return action, flag

        if flag == False:

            if np.random.uniform() < self.epsilon:  # 375个动作里随机选Q值最大的一个动作
                # forward feed the observation and get q value for every actions,向前输出 这里使用的是估计神经网络
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation[:]})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)  # 375个动作里随机选一个
            return action, flag

    # --------------------------------------学习过程------------------------------------------
    # 根据当前状态s，先去evaluate网络找到s'对应Q值最大的动作a'
    # 再去target找到对应该动作a'的Q值作为target值，Q(s', argmax_a'(s',a';θi),θi')

    def learn(self):

        # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # 从记忆库中提取记忆

        # 从 memory=500 中随机抽取 batch_size=50 这么多记忆
        # 它是一个batch一个batch的训练的 不是说每一步就训练哪一步
        if self.memory_counter > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        # 构成批记忆库
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params，后32
                self.s: batch_memory[:, :self.n_features],  # newest params,前32
            })

        # 给target网络输入s'以获得下一个最优a'，三维（1，50，375）
        q_next_eval = self.sess.run(
            [self.q_eval],
            feed_dict={
                self.s: batch_memory[:, -self.n_features:],
            })
        q_next_eval = np.array(q_next_eval).reshape(50, 275)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)  # [0 1 2 3 4 ...  48 49]

        eval_act_index = batch_memory[:, self.n_features].astype(int)  # 对应所采取的动作

        reward = batch_memory[:, self.n_features + 1]  # 对应的奖励
        # print("reward type:", type(reward))

        q_maxindex = q_next_eval.argmax(axis=1)  # 找到evaluate网络下s'对应的Q值最大的动作索引值

        maxvalue = []

        for i in range(50):
            for j in q_maxindex:
                maxvalue.append(q_next[i, j])
                break
        # maxvalue = np.array(maxvalue).reshape(50, 1)
        # print("maxvalue: ", type(maxvalue))

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.array(maxvalue[:])

        # 训练eval网络
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        # print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        x = []
        y = []
        for i in range(len(self.cost_his)):
            if i % 50 == 0:
                x.append(np.arange(len(self.cost_his))[i])
                y.append(self.cost_his[i])
        plt.plot(x, y, marker='o', label='learning_rate=0.01')

        plt.legend(loc=1)
        plt.ylabel('Loss')
        plt.xlabel('Training steps')
        plt.savefig('Loss_DDQN_GMR_USER_100000.eps', bbox_inches='tight')
        plt.show()


start = time.time()

N = 5000  # 回合次数


# 主程序
def run_maze():
    return1 = []
    average_return1 = []

    migration1 = []
    episode_migration1 = []

    cumulative_reward1 = 0
    m1 = 0

    step = 0

    for episode in range(N):

        U = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 10个用户累计迁移量
        T = 0
        m = 0
        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action, Flag = RL.choose_action(observation, U)  # 选动作

            observation_, reward, Reward, done, u = env.step(observation, action, Flag, T, U)  # 状态更新，包括用户移动更新

            cumulative_reward1 += reward  # 累计全部奖励

            m1 += u  # 记录全部迁移量
            m += u  # 记录每回合的迁移量

            RL.store_transition(observation, action, Reward, observation_)

            if (step > 400) and done:
                RL.learn()

            observation = observation_

            step += 1
            T += 1

            if done:
                return1.append(cumulative_reward1)  # 总奖励
                average_return1.append(cumulative_reward1 / step)  # 平均奖励

                migration1.append(m1)  # 总吞吐量
                episode_migration1.append(m)  # 回合吞吐量

                print("回合：", episode)
                print("每个用户卸载迁移量", U)
                print("回合卸载迁移量", np.sum(U))

                break

    print("================平均奖励=======================", np.mean(average_return1))
    print("================平均回合吞吐量==================", np.mean(episode_migration1))

    env.reset_uav()
    env.render()
    end = time.time()
    print("game over!")
    print('运行时间:', end - start)

    # env.destory()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 尝试减少替换次数
                      memory_size=2000,  # 尝试扩大记忆库
                      output_graph=False
                      )

    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()