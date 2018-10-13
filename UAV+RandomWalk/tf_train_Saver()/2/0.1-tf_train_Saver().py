# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/18


import os

import tensorflow as tf

# Create some variables.

v1 = tf.Variable([[1, 1], [2, 2], [3, 3]], name="v1")

v2 = tf.Variable([[4, 4], [5, 5], [6, 7]], name="v2")

# Add an op to initialize the variables.

init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

# 判断模型保存路径是否存在，不存在就创建
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

# Later, launch the model, initialize the variables, do some work, save the variables to disk.

with tf.Session() as sess:

    if os.path.exists('tmp/checkpoint'):          # 判断模型是否存在
        saver.restore(sess, 'tmp/model.ckpt')     # 存在就从模型中恢复变量
    else:
        init = tf.global_variables_initializer()  # 不存在就初始化变量
        sess.run(init_op)

    # Do some work with the model.

    save_path = saver.save(sess, "/tmp/model.ckpt")

    print("Model saved in file: ", save_path)