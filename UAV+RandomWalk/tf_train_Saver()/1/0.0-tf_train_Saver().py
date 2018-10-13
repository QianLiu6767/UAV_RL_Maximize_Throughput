# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/9/17


import tensorflow as tf
import numpy as np
import os

# 用numpy产生数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 转置
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 输入层
x_ph = tf.placeholder(tf.float32, [None, 1])
y_ph = tf.placeholder(tf.float32, [None, 1])

# 隐藏层
w1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
wx_plus_b1 = tf.matmul(x_ph, w1) + b1
hidden = tf.nn.relu(wx_plus_b1)

# 输出层
w2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]) + 0.1)
wx_plus_b2 = tf.matmul(hidden, w2) + b2
y = wx_plus_b2

# 损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ph - y), reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 保存模型对象saver，默认保存最近5次模型，max_to_keep=0保存所有模型/1仅保存最近1次
saver = tf.train.Saver()

# 判断模型保存路径是否存在，不存在就创建
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

# 初始化
with tf.Session() as sess:
    if os.path.exists('tmp/checkpoint'):          # 判断模型是否存在
        saver.restore(sess, 'tmp/model.ckpt')     # 存在就从模型中恢复变量

        _, loss_value = sess.run([train_op, loss], feed_dict={x_ph: x_data, y_ph: y_data})
        yy = []
        yy.append(sess.run(y, feed_dict={x_ph: x_data, y_ph: y_data}))

        print("调用模型-y真实值，y预测值：\n", y_data, yy)
        print("调用模型-损失:", loss_value)


    else:
        init = tf.global_variables_initializer()  # 不存在就初始化变量
        sess.run(init)

        for i in range(1000):
            _, loss_value = sess.run([train_op, loss], feed_dict={x_ph: x_data, y_ph: y_data})

            if (i%50 == 0):
                save_path = saver.save(sess, 'tmp/model.ckpt')
                print("训练模型-迭代次数：%d , 训练损失：%s" % (i, loss_value))