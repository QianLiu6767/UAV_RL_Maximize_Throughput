# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/5/24

import numpy as np

d = []
for i in range(15):

    a = np.array((np.random.rand(1) * 1000))

    b = np.array((np.random.rand(1) * 1000))

    c = []
    c.append(int(a))
    c.append(int(b))

    d.append(c)

print(d[13])
print(d[13][0])
