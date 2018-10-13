# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/5

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import deque
from random import sample


class ReplayMemory:
    def __init__(self, capacity):
        self.samples = deque([], maxlen=capacity)

    def store(self, exp):
        self.samples.append(exp)
        pass

    def get_batch(self, n):
        n_samples = min(n, len(self.samples))
        samples = sample(self.samples, n_samples)
        return samples
