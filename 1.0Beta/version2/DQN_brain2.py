# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/5/20

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork1:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net1'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net1'):
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

    def store_transition1(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,通过状态算出动作值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # -------------------------从记忆库中提取记忆--------------------------
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # 构成批记忆库
        batch_memory = self.memory[sample_index, :]

        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],  # 前n_features列的数据，表示方法很叼
                self.a: batch_memory[:, self.n_features],  # 第n_features列的数据
                self.r: batch_memory[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his.append(cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt

        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.tick_params(labelsize=20)

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training steps', font2)
        """
        x = []
        y = []
        for i in range(len(self.cost_his)):
            if i % 40 == 0:
                x.append(np.arange(len(self.cost_his))[i])
                y.append(self.cost_his[i])
        plt.plot(x, y, marker='o', label='learning_rate=0.09', markersize=10)

        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20}
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.legend(loc=1)
        plt.legend(prop=font1, edgecolor='black', facecolor='white')

        plt.tick_params(labelsize=20)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training episodes', font2)
        # plt.show()
        """


class DeepQNetwork2:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory_ = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his_ = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net2'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net2'):
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

    def store_transition2(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter_'):
            self.memory_counter_ = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter_ % self.memory_size
        self.memory_[index, :] = transition
        self.memory_counter_ += 1

    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,通过状态算出动作值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # -------------------------从记忆库中提取记忆--------------------------
        # sample batch memory from all memory
        if self.memory_counter_ > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter_, size=self.batch_size)
        # 构成批记忆库
        batch_memory_ = self.memory_[sample_index, :]

        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory_[:, :self.n_features],  # 前n_features列的数据，表示方法很叼
                self.a: batch_memory_[:, self.n_features],  # 第n_features列的数据
                self.r: batch_memory_[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory_[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his_.append(cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        x_ = []
        y_ = []
        for i in range(len(self.cost_his_)):
            if i % 40 == 0:
                x_.append(np.arange(len(self.cost_his_))[i])
                y_.append(self.cost_his_[i])
        plt.plot(x_, y_, marker='^', label='learning_rate=0.05', markersize=10)
        #plt.plot(np.arange(len(self.cost_his_)), self.cost_his_)
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20}
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.legend(loc=1)
        plt.legend(prop=font1, edgecolor='black', facecolor='white')

        plt.tick_params(labelsize=20)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training episodes', font2)
        #plt.show()


class DeepQNetwork3:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory__ = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his__ = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net3'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net3'):
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

    def store_transition3(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter__ = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter__ % self.memory_size
        self.memory__[index, :] = transition
        self.memory_counter__ += 1

    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,通过状态算出动作值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # -------------------------从记忆库中提取记忆--------------------------
        # sample batch memory from all memory
        if self.memory_counter__ > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter__, size=self.batch_size)
        # 构成批记忆库
        batch_memory__ = self.memory__[sample_index, :]

        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory__[:, :self.n_features],  # 前n_features列的数据，表示方法很叼
                self.a: batch_memory__[:, self.n_features],  # 第n_features列的数据
                self.r: batch_memory__[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory__[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his__.append(cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        x__ = []
        y__ = []
        for i in range(len(self.cost_his__)):
            if i % 40 == 0:
                x__.append(np.arange(len(self.cost_his__))[i])
                y__.append(self.cost_his__[i])
        plt.plot(x__, y__, marker='x', label='learning_rate=0.01', markersize=10)
        #plt.plot(np.arange(len(self.cost_his__)), self.cost_his__)
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20}
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.legend(loc=1)
        plt.legend(prop=font1, edgecolor='black', facecolor='white')

        plt.tick_params(labelsize=20)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training episodes', font2)
        #plt.show()


class DeepQNetwork4:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory__ = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his__ = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net4'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net4'):
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

    def store_transition4(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter__ = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter__ % self.memory_size
        self.memory__[index, :] = transition
        self.memory_counter__ += 1

    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,通过状态算出动作值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # -------------------------从记忆库中提取记忆--------------------------
        # sample batch memory from all memory
        if self.memory_counter__ > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter__, size=self.batch_size)
        # 构成批记忆库
        batch_memory__ = self.memory__[sample_index, :]

        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory__[:, :self.n_features],  # 前n_features列的数据，表示方法很叼
                self.a: batch_memory__[:, self.n_features],  # 第n_features列的数据
                self.r: batch_memory__[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory__[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his__.append(cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        x__ = []
        y__ = []
        for i in range(len(self.cost_his__)):
            if i % 40 == 0:
                x__.append(np.arange(len(self.cost_his__))[i])
                y__.append(self.cost_his__[i])
        plt.plot(x__, y__, marker='x', label='learning_rate=0.01', markersize=10)
        #plt.plot(np.arange(len(self.cost_his__)), self.cost_his__)
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20}
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.legend(loc=1)
        plt.legend(prop=font1, edgecolor='black', facecolor='white')

        plt.tick_params(labelsize=20)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training episodes', font2)
        #plt.show()


class DeepQNetwork5:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory__ = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his__ = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net5'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net5'):
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

    def store_transition5(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter__ = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter__ % self.memory_size
        self.memory__[index, :] = transition
        self.memory_counter__ += 1

    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,通过状态算出动作值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # -------------------------从记忆库中提取记忆--------------------------
        # sample batch memory from all memory
        if self.memory_counter__ > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter__, size=self.batch_size)
        # 构成批记忆库
        batch_memory__ = self.memory__[sample_index, :]

        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory__[:, :self.n_features],  # 前n_features列的数据，表示方法很叼
                self.a: batch_memory__[:, self.n_features],  # 第n_features列的数据
                self.r: batch_memory__[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory__[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his__.append(cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        x__ = []
        y__ = []
        for i in range(len(self.cost_his__)):
            if i % 40 == 0:
                x__.append(np.arange(len(self.cost_his__))[i])
                y__.append(self.cost_his__[i])
        plt.plot(x__, y__, marker='x', label='learning_rate=0.01', markersize=10)
        #plt.plot(np.arange(len(self.cost_his__)), self.cost_his__)
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20}
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.legend(loc=1)
        plt.legend(prop=font1, edgecolor='black', facecolor='white')

        plt.tick_params(labelsize=20)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training episodes', font2)
        #plt.show()


class DeepQNetwork6:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory__ = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his__ = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net6'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net6'):
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

    def store_transition6(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter__ = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter__ % self.memory_size
        self.memory__[index, :] = transition
        self.memory_counter__ += 1

    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,通过状态算出动作值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # -------------------------从记忆库中提取记忆--------------------------
        # sample batch memory from all memory
        if self.memory_counter__ > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter__, size=self.batch_size)
        # 构成批记忆库
        batch_memory__ = self.memory__[sample_index, :]

        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory__[:, :self.n_features],  # 前n_features列的数据，表示方法很叼
                self.a: batch_memory__[:, self.n_features],  # 第n_features列的数据
                self.r: batch_memory__[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory__[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his__.append(cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        x__ = []
        y__ = []
        for i in range(len(self.cost_his__)):
            if i % 40 == 0:
                x__.append(np.arange(len(self.cost_his__))[i])
                y__.append(self.cost_his__[i])
        plt.plot(x__, y__, marker='x', label='learning_rate=0.01', markersize=10)
        #plt.plot(np.arange(len(self.cost_his__)), self.cost_his__)
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20}
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.legend(loc=1)
        plt.legend(prop=font1, edgecolor='black', facecolor='white')

        plt.tick_params(labelsize=20)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training episodes', font2)
        #plt.show()


class DeepQNetwork7:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory__ = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his__ = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net7'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net7'):
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

    def store_transition7(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter__ = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter__ % self.memory_size
        self.memory__[index, :] = transition
        self.memory_counter__ += 1

    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,通过状态算出动作值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # -------------------------从记忆库中提取记忆--------------------------
        # sample batch memory from all memory
        if self.memory_counter__ > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter__, size=self.batch_size)
        # 构成批记忆库
        batch_memory__ = self.memory__[sample_index, :]

        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory__[:, :self.n_features],  # 前n_features列的数据，表示方法很叼
                self.a: batch_memory__[:, self.n_features],  # 第n_features列的数据
                self.r: batch_memory__[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory__[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his__.append(cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        x__ = []
        y__ = []
        for i in range(len(self.cost_his__)):
            if i % 40 == 0:
                x__.append(np.arange(len(self.cost_his__))[i])
                y__.append(self.cost_his__[i])
        plt.plot(x__, y__, marker='x', label='learning_rate=0.01', markersize=10)
        #plt.plot(np.arange(len(self.cost_his__)), self.cost_his__)
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20}
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.legend(loc=1)
        plt.legend(prop=font1, edgecolor='black', facecolor='white')

        plt.tick_params(labelsize=20)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training episodes', font2)
        #plt.show()


class DeepQNetwork8:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions  # 动作个数
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory__ = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his__ = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net8'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net8'):
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

    def store_transition8(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter__ = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter__ % self.memory_size
        self.memory__[index, :] = transition
        self.memory_counter__ += 1

    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        print('observation: ', observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions,通过状态算出动作值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        print(action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # -------------------------从记忆库中提取记忆--------------------------
        # sample batch memory from all memory
        if self.memory_counter__ > self.memory_size:
            # memory_size=2000,batch_size=32，从2000中随机选择32个数当作采样索引
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter__, size=self.batch_size)
        # 构成批记忆库
        batch_memory__ = self.memory__[sample_index, :]

        # -------------------------用批记忆库训练神经网络-----------------------
        # 训练eval_net
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory__[:, :self.n_features],  # 前n_features列的数据，表示方法很叼
                self.a: batch_memory__[:, self.n_features],  # 第n_features列的数据
                self.r: batch_memory__[:, self.n_features + 1],  # 第n_features+1列的数据
                self.s_: batch_memory__[:, -self.n_features:],   # 最后n_features列的数据
            })

        self.cost_his__.append(cost)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning_step_counter:', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        x__ = []
        y__ = []
        for i in range(len(self.cost_his__)):
            if i % 40 == 0:
                x__.append(np.arange(len(self.cost_his__))[i])
                y__.append(self.cost_his__[i])
        plt.plot(x__, y__, marker='x', label='learning_rate=0.01', markersize=10)
        #plt.plot(np.arange(len(self.cost_his__)), self.cost_his__)
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20}
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.legend(loc=1)
        plt.legend(prop=font1, edgecolor='black', facecolor='white')

        plt.tick_params(labelsize=20)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22}
        plt.ylabel('Loss', font2)
        plt.xlabel('Training episodes', font2)
        #plt.show()


if __name__ == '__main__':
    DQN = DeepQNetwork1(3, 4, output_graph=False)
