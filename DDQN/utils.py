# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/5

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np


def prep_input(data, n_dimension):
    prep = np.asarray(data)
    transformed = prep.reshape((1, n_dimension))
    return transformed


def prep_batch(to_prep):
    prep = np.vstack(to_prep)
    return prep
