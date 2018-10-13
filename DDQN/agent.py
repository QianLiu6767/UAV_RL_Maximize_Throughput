# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/5

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from memory import ReplayMemory
from observer import EpsilonUpdater
from parameters import *
from qnet import NN


class DQNAgent:
    def __init__(self, environment):
        self.env = environment
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.dim_actions = self.env.action_space.n
        self.dim_states = self.env.observation_space.shape
        self.NN = NN(self.env.observation_space.shape, self.env.action_space.n,
                     BATCH_SIZE, SIZE_HIDDEN, LEARNING_RATE, ACTIVATION)
        self.observers = []
        self.episode_count = 0
        self.step_count_total = 1
        self.step_count_episode = 1
        self.epsilon_min = EPSILON_MIN
        self.epsilon_max = EPSILON_MAX
        self.epsilon_decay = EPSILON_DECAY
        self.target_update = TARGET_UPDATE
        self.max_steps = MAX_STEPS
        self.n_episodes = N_EPISODES
        self.epsilon = EPSILON_MAX
        self.batch_size = BATCH_SIZE
        self.usetarget = False
        self.gamma = GAMMA
        self.loss = 0
        self.done = False
        self.reward = 0
        self.reward_episode = 0
        self.learning_switch = False
        self.learning_start = LEARNING_START

    def notify(self, event):
        for observer in self.observers:
            observer(event)
        pass

    def act(self, state):
        self.step_count_total += 1
        action = self.choose_action(state)
        return action

    def learn(self, obs):
        self.memory.store(obs)
        if self.learning_switch:
            self.backup()
        self.notify('step_done')
        pass

    def backup(self):
        self.flashback()
        if self.step_count_total % self.target_update == 0:
            print('update')
            print(self.epsilon)
            self.NN.update_target()
            self.usetarget = True
        pass

    def flashback(self):
        X, y = self._make_batch()
        self.loss = self.NN.train(X, y)
        if np.isnan(self.loss.history['loss']).any():
            print('Warning, loss is {}'.format(self.loss))
        pass

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            choice = self.random_choice()
        else:
            choice = self.greedy_choice(state)
        return choice

    def greedy_choice(self, state):
        greedy_choice = self.NN.best_action(state, usetarget=False)
        return greedy_choice

    def random_choice(self):
        random_choice = np.random.randint(0, self.dim_actions)
        return random_choice

    def _make_batch(self):
        X = []
        y = []
        batch = self.memory.get_batch(self.batch_size)
        for state, action, newstate, reward, done in batch:
            X.append(state)
            target = self.NN.predict(state, False)
            q_vals_new_t = self.NN.predict(newstate, self.usetarget)
            a_select = self.NN.best_action(newstate, False)
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * q_vals_new_t[a_select]
            y.append(target)
        return X, y

    def add_observer(self, observer):
        self.observers.append(observer)
        pass
