# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/5

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import deque

import agent
import gym
import observer
from parameters import *


class Experiment:
    def __init__(self, environment):
        self.env = gym.make(environment)
        self.episode_count = 0
        self.reward_buffer = deque([], maxlen=100)

    def run_experiment(self, agent):
        self.env.monitor.start('/tmp/cartpole', force=True)
        for n in range(N_EPISODES):
            self.run_episode(agent)
        self.env.monitor.close()
        pass

    def run_episode(self, agent):
        self.reward = 0
        s = self.env.reset()
        done = False
        while not done:
            self.env.render()
            a = agent.act(s)
            s_, r, done, _ = self.env.step(a)
            agent.learn((s, a, s_, r, done))
            self.reward += r
            s = s_

        self.episode_count += 1
        self.reward_buffer.append(self.reward)
        average = sum(self.reward_buffer) / len(self.reward_buffer)

        print("Episode Nr. {} \nScore: {} \nAverage: {}".format(
            self.episode_count, self.reward, average))


if __name__ == "__main__":
    import gym
    import agent
    import observer
    # observer
    key = 'CartPole-v0'
    exp = Experiment(key)
    agent = agent.DQNAgent(exp.env)
    epsilon = observer.EpsilonUpdater(agent)
    agent.add_observer(epsilon)
    exp.run_experiment(agent)

    #epsilon = observer.EpsilonUpdater(agent)
    #agent.add_observer(epsilon)
