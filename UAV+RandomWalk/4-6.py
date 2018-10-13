# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/3

# DQN + 高斯马尔科夫移动模型

import numpy as np
import tkinter as tk
import time
import tensorflow as tf
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
        self.title('UAV测试环境-GMR')
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

        for j in range(15):
            self.canvas.create_oval(max(0, self.user_center[j, 0] - 5), max(self.user_center[j, 1] - 5, 0),
                                        min(1000, self.user_center[j, 0] + 5), min(self.user_center[j, 1] + 5, 600),
                                        fill='red')

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
        for j in range(15):
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
                                  (s[1] - self.oval_center[j, 1]) ** 2)) / 800

                # 盘旋能耗
                distance = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                for k in range(15):
                    distance[k] = (next_coords[0] - self.user_center[k, 0])**2 +\
                                  (next_coords[1] - self.user_center[k, 1])**2

                o = distance.tolist().index(distance.min())

                # 盘旋能耗
                eh = p * (u[o] / np.emath.log2(1 + ((p*(rho / (100**2 + distance.min())))/sigma))) * 5

                # 计算能耗
                ec = 0.3 * u[o] / 10

                # 效用函数
                utility = 1 - np.exp(-((u[o] ** 2) / (u[o] + 10)))

                self.battery -= (ef + ec + eh)

                # 奖励
                reward = utility - ef - ec - eh

                if self.battery <= p*(np.sqrt((40-self.oval_center[j, 0])**2+(40-self.oval_center[j, 1])**2))/800:
                    done = True
                else:
                    done = False
                # s_ = np.array([next_coords[0] / (MAZE_H * UNIT), next_coords[1] / (MAZE_W * UNIT)])

                # 下一个状态
                s_ = np.hstack((np.array([next_coords[0] / (MAZE_H * UNIT), next_coords[1] / (MAZE_W * UNIT)]),
                                self.battery / 10))
                return s_, reward, done, u[o]

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
        self.n_actions = n_actions                        # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0                                   # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 一共学习了多少步
        self.learn_step_counter = 0

        # 初始化记忆库 [s,a,r,s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # 建立神经网络,两个：evaluate 和 target
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

    """
    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            # axis=1,每行的最大值
            # q_target = (1 - self.gamma) * self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')

            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            # tf.stack拼接矩阵,
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    """

    def _build_net(self):
        # -------------- 创建 eval 神经网络, 及时提升参数 --------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions],
                                       name='Q_target')  # 用来接收 q_target 的值, 这个之后会通过计算得到
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

        with tf.variable_scope('loss'):  # 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):  # 梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
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

        saver = tf.train.Saver()

    # 储存记忆，从上往下储存，满了就循环
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 选择动作，90%使用DQN选择，10%使用随机办法
    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,向前输出 这里使用的是估计神经网络
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    # 学习过程
    def learn(self):

        # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # --------------------------从记忆库中提取记忆--------------------------

        # 从 memory 中随机抽取 batch_size 这么多记忆
        # 它是一个batch一个batch的训练的 不是说每一步就训练哪一步
        if self.memory_counter > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        # 构成批记忆库
        batch_memory = self.memory[sample_index, :]

        """
        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],     # 前n_features列的数据，表示方法很叼
                self.a: batch_memory[:, self.n_features],      # 第n_features列的数据
                self.r: batch_memory[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his.append(cost)
        """
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],    # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 训练eval网络
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

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
        plt.show()


start = time.time()

# 回合次数
N = 100000


# 主程序
def run_maze():
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []

    migration1 = []
    average_migration1 = []

    m1 = 0
    step = 0

    for episode in range(N):

        V = [10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20]
        W = [30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 330, 360]
        T = 0

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            env.update_env(V, W, T)

            action = RL.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m1 += u                     # 记录每一步的迁移量

            RL.store_transition(observation, action, reward, observation_)

            if (step > 400) and done:
                RL.learn()

            observation = observation_

            step += 1
            T += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration1.append(m1)
                average_migration1.append(m1 / step)
                cumulative_reward1_ = 0
                break

    # 相同学习率，不同电量的平均吞吐量对比图
    plt.figure(figsize=(10, 8))
    # plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    x1 = []

    g1 = []

    for i in range(len(average_migration1)):
        if i % 1 == 0:
            x1.append(np.arange(len(average_migration1))[i])
            g1.append(average_migration1[i])

    plt.plot(x1, g1, marker='o', label='BatteryLevel1', markersize=10)

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
    plt.savefig('a_t_DQN_100000.eps', bbox_inches='tight')
    plt.show()

    # 相同学习率，不同电量的平均奖励对比图
    plt.figure(figsize=(10, 8))
    x1 = []
    r1 = []

    for i in range(len(average_return1)):
        if i % 1 == 0:
            x1.append(np.arange(len(average_return1))[i])
            r1.append(average_return1[i])

    plt.plot(x1, r1, marker='o', label='BatteryLevel1', markersize=10)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    plt.legend(loc="lower right")

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}

    plt.ylabel('Average Return', font2)
    plt.xlabel('Training Episodes', font2)
    plt.savefig('a_r_DQN_1000000.eps', bbox_inches='tight')
    plt.show()

    """
    # 回合奖励
    # plt.figure(3)
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(len(episode_return1)), episode_return1, markersize=10)
    plt.plot(np.arange(len(episode_return2)), episode_return2, markersize=10)
    plt.plot(np.arange(len(episode_return3)), episode_return3, markersize=10)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    # plt.legend(loc=4)

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}

    plt.ylabel('Episode_Return'), font2
    plt.xlabel('Training episodes', font2)
    plt.show()


    # 回合奖励的采样图
    plt.figure(4)
    x4 = []
    y4 = []
    for i in range(len(episode_return1)):
        if i % 50 == 0:
            x4.append(np.arange(len(episode_return1))[i])
            y4.append(episode_return1[i])

    plt.plot(x4, y4, marker='o')

    plt.show()
    """

    env.reset_uav()
    env.render()
    end = time.time()
    print("game over!")
    print('运行时间:', end - start)
    engine = pyttsx3.init()                         # 文字转语音模块
    engine.say('程序运行完成')
    engine.runAndWait()
    # env.destory()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,       # 尝试减少替换次数
                      memory_size=2000,              # 尝试扩大记忆库
                      output_graph=False
                      )

    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
