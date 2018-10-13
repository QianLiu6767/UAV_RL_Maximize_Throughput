# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/5

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import exp


class EpsilonUpdater:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, event):
        if event == 'step_done':
            self.epsilon_update()
            self.switch_learning()
        else:
            pass

    def epsilon_update(self):
        self.agent.epsilon = (
            self.agent.epsilon_min +
            (self.agent.epsilon_max - self.agent.epsilon_min) * exp(
                -self.agent.epsilon_decay * self.agent.step_count_total))
        pass

    def switch_learning(self):
        if self.agent.step_count_total >= self.agent.learning_start:
            self.agent.learning_switch = True
        pass
