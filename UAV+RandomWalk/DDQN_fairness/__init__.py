# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/11/2

import numpy as np

speed =[np.random.uniform(0.5, 1, 1),]
angle = [np.random.randint(0, 360, 1),]
a = []
for i in range(100):
    speed.append(0.99 * speed[-1] + (1 - 0.99) * 2 + (1 - 0.99) ** 2 * np.random.normal(0, 0.1, 1))
    angle.append(0.9 * angle[-1] + (1 - 0.9) * 360 + (1 - 0.9) ** 2 * np.random.normal(180, 180, 1))
    a.append(np.random.normal(180, 20, 1))


print("速度变化", speed)
print("角度变化", angle)
print(a)