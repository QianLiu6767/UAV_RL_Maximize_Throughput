# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/5/20

# 训练不同大小记忆库对平均吞吐量的影响
#
from Maze2 import Maze
from DQN_brain2 import DeepQNetwork1
from DQN_brain2 import DeepQNetwork2
from DQN_brain2 import DeepQNetwork3
from DQN_brain2 import DeepQNetwork4
from DQN_brain2 import DeepQNetwork5
from DQN_brain2 import DeepQNetwork6
from DQN_brain2 import DeepQNetwork7
from DQN_brain2 import DeepQNetwork8
import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3

N = 1000
start = time.time()

#  运行前是否调整epsilon


def run_maze():
    # ---------------------第一次--------------------
    # RL1
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m1 = 0
    step = 0
    migration1 = []
    average_migration1 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL1.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m1 += u  # 记录每一步的迁移量

            RL1.store_transition1(observation, action, reward, observation_)

            if (step > 400) and done:
                RL1.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration1.append(m1)
                average_migration1.append(m1 / step)
                cumulative_reward1_ = 0
                break

    # RL2
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m2 = 0
    step = 0
    migration2 = []
    average_migration2 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL2.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m2 += u  # 记录每一步的迁移量

            RL2.store_transition2(observation, action, reward, observation_)

            if (step > 400) and done:
                RL2.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration2.append(m2)
                average_migration2.append(m2 / step)
                cumulative_reward1_ = 0
                break
    # RL3
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m3 = 0
    step = 0
    migration3 = []
    average_migration3 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL3.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m3 += u  # 记录每一步的迁移量

            RL3.store_transition3(observation, action, reward, observation_)

            if (step > 400) and done:
                RL3.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration3.append(m3)
                average_migration3.append(m3 / step)
                cumulative_reward1_ = 0
                break
    # RL4
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m4 = 0
    step = 0
    migration4 = []
    average_migration4 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL4.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m4 += u  # 记录每一步的迁移量

            RL4.store_transition4(observation, action, reward, observation_)

            if (step > 400) and done:
                RL4.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration4.append(m4)
                average_migration4.append(m4 / step)
                cumulative_reward1_ = 0
                break
    # RL5
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m5 = 0
    step = 0
    migration5 = []
    average_migration5 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL5.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m5 += u  # 记录每一步的迁移量

            RL5.store_transition5(observation, action, reward, observation_)

            if (step > 400) and done:
                RL5.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration5.append(m5)
                average_migration5.append(m5 / step)
                cumulative_reward1_ = 0
                break
    # RL6
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m6 = 0
    step = 0
    migration6 = []
    average_migration6 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL6.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m6 += u  # 记录每一步的迁移量

            RL6.store_transition6(observation, action, reward, observation_)

            if (step > 400) and done:
                RL6.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration6.append(m6)
                average_migration6.append(m6 / step)
                cumulative_reward1_ = 0
                break
    # RL7
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m7 = 0
    step = 0
    migration7 = []
    average_migration7 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL7.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m7 += u  # 记录每一步的迁移量

            RL7.store_transition7(observation, action, reward, observation_)

            if (step > 400) and done:
                RL7.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration7.append(m7)
                average_migration7.append(m7 / step)
                cumulative_reward1_ = 0
                break
    # RL8
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m8 = 0
    step = 0
    migration8 = []
    average_migration8 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL8.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m8 += u  # 记录每一步的迁移量

            RL8.store_transition8(observation, action, reward, observation_)

            if (step > 400) and done:
                RL8.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration8.append(m8)
                average_migration8.append(m8 / step)
                cumulative_reward1_ = 0
                break

    data_1 = [sum( average_migration1)/len( average_migration1), sum( average_migration2)/len( average_migration2),
              sum( average_migration3)/len( average_migration3), sum( average_migration4)/len( average_migration4),
              sum( average_migration5)/len( average_migration5), sum( average_migration6)/len( average_migration6),
              sum( average_migration7)/len( average_migration7), sum( average_migration8)/len( average_migration8)]
    # data_1 = [max(average_return1), max(average_return2), max(average_return3), max(average_return4),
              # max(average_return5), max(average_return6), max(average_return7), max(average_return8)]

    # -------------------------第二次---------------------
    # RL1
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m1 = 0
    step = 0
    migration1 = []
    average_migration1 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL1.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m1 += u  # 记录每一步的迁移量

            RL1.store_transition1(observation, action, reward, observation_)

            if (step > 400) and done:
                RL1.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration1.append(m1)
                average_migration1.append(m1 / step)
                cumulative_reward1_ = 0
                break

    # RL2
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m2 = 0
    step = 0
    migration2 = []
    average_migration2 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL2.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m2 += u  # 记录每一步的迁移量

            RL2.store_transition2(observation, action, reward, observation_)

            if (step > 400) and done:
                RL2.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration2.append(m2)
                average_migration2.append(m2 / step)
                cumulative_reward1_ = 0
                break
    # RL3
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m3 = 0
    step = 0
    migration3 = []
    average_migration3 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL3.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m3 += u  # 记录每一步的迁移量

            RL3.store_transition3(observation, action, reward, observation_)

            if (step > 400) and done:
                RL3.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration3.append(m3)
                average_migration3.append(m3 / step)
                cumulative_reward1_ = 0
                break
    # RL4
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m4 = 0
    step = 0
    migration4 = []
    average_migration4 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL4.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m4 += u  # 记录每一步的迁移量

            RL4.store_transition4(observation, action, reward, observation_)

            if (step > 400) and done:
                RL4.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration4.append(m4)
                average_migration4.append(m4 / step)
                cumulative_reward1_ = 0
                break
    # RL5
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m5 = 0
    step = 0
    migration5 = []
    average_migration5 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL5.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m5 += u  # 记录每一步的迁移量

            RL5.store_transition5(observation, action, reward, observation_)

            if (step > 400) and done:
                RL5.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration5.append(m5)
                average_migration5.append(m5 / step)
                cumulative_reward1_ = 0
                break
    # RL6
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m6 = 0
    step = 0
    migration6 = []
    average_migration6 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL6.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m6 += u  # 记录每一步的迁移量

            RL6.store_transition6(observation, action, reward, observation_)

            if (step > 400) and done:
                RL6.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration6.append(m6)
                average_migration6.append(m6 / step)
                cumulative_reward1_ = 0
                break
    # RL7
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m7 = 0
    step = 0
    migration7 = []
    average_migration7 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL7.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m7 += u  # 记录每一步的迁移量

            RL7.store_transition7(observation, action, reward, observation_)

            if (step > 400) and done:
                RL7.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration7.append(m7)
                average_migration7.append(m7 / step)
                cumulative_reward1_ = 0
                break
    # RL8
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m8 = 0
    step = 0
    migration8 = []
    average_migration8 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL8.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m8 += u  # 记录每一步的迁移量

            RL8.store_transition8(observation, action, reward, observation_)

            if (step > 400) and done:
                RL8.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration8.append(m8)
                average_migration8.append(m8 / step)
                cumulative_reward1_ = 0
                break

    data_2 = [sum(average_migration1) / len(average_migration1), sum(average_migration2) / len(average_migration2),
              sum(average_migration3) / len(average_migration3), sum(average_migration4) / len(average_migration4),
              sum(average_migration5) / len(average_migration5), sum(average_migration6) / len(average_migration6),
              sum(average_migration7) / len(average_migration7), sum(average_migration8) / len(average_migration8)]
    # data_2 = [max(average_return1), max(average_return2), max(average_return3), max(average_return4),
    # max(average_return5), max(average_return6), max(average_return7), max(average_return8)]

    # -----------------------第三次----------------------
    # RL1
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m1 = 0
    step = 0
    migration1 = []
    average_migration1 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL1.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m1 += u  # 记录每一步的迁移量

            RL1.store_transition1(observation, action, reward, observation_)

            if (step > 400) and done:
                RL1.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration1.append(m1)
                average_migration1.append(m1 / step)
                cumulative_reward1_ = 0
                break

    # RL2
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m2 = 0
    step = 0
    migration2 = []
    average_migration2 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL2.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m2 += u  # 记录每一步的迁移量

            RL2.store_transition2(observation, action, reward, observation_)

            if (step > 400) and done:
                RL2.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration2.append(m2)
                average_migration2.append(m2 / step)
                cumulative_reward1_ = 0
                break
    # RL3
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m3 = 0
    step = 0
    migration3 = []
    average_migration3 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL3.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m3 += u  # 记录每一步的迁移量

            RL3.store_transition3(observation, action, reward, observation_)

            if (step > 400) and done:
                RL3.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration3.append(m3)
                average_migration3.append(m3 / step)
                cumulative_reward1_ = 0
                break
    # RL4
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m4 = 0
    step = 0
    migration4 = []
    average_migration4 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL4.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m4 += u  # 记录每一步的迁移量

            RL4.store_transition4(observation, action, reward, observation_)

            if (step > 400) and done:
                RL4.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration4.append(m4)
                average_migration4.append(m4 / step)
                cumulative_reward1_ = 0
                break
    # RL5
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m5 = 0
    step = 0
    migration5 = []
    average_migration5 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL5.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m5 += u  # 记录每一步的迁移量

            RL5.store_transition5(observation, action, reward, observation_)

            if (step > 400) and done:
                RL5.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration5.append(m5)
                average_migration5.append(m5 / step)
                cumulative_reward1_ = 0
                break
    # RL6
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m6 = 0
    step = 0
    migration6 = []
    average_migration6 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL6.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m6 += u  # 记录每一步的迁移量

            RL6.store_transition6(observation, action, reward, observation_)

            if (step > 400) and done:
                RL6.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration6.append(m6)
                average_migration6.append(m6 / step)
                cumulative_reward1_ = 0
                break
    # RL7
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m7 = 0
    step = 0
    migration7 = []
    average_migration7 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL7.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m7 += u  # 记录每一步的迁移量

            RL7.store_transition7(observation, action, reward, observation_)

            if (step > 400) and done:
                RL7.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration7.append(m7)
                average_migration7.append(m7 / step)
                cumulative_reward1_ = 0
                break
    # RL8
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    m8 = 0
    step = 0
    migration8 = []
    average_migration8 = []

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL8.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m8 += u  # 记录每一步的迁移量

            RL8.store_transition8(observation, action, reward, observation_)

            if (step > 400) and done:
                RL8.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration8.append(m8)
                average_migration8.append(m8 / step)
                cumulative_reward1_ = 0
                break

    data_3 = [sum(average_migration1) / len(average_migration1), sum(average_migration2) / len(average_migration2),
              sum(average_migration3) / len(average_migration3), sum(average_migration4) / len(average_migration4),
              sum(average_migration5) / len(average_migration5), sum(average_migration6) / len(average_migration6),
              sum(average_migration7) / len(average_migration7), sum(average_migration8) / len(average_migration8)]
    # data_3 = [max(average_return1), max(average_return2), max(average_return3), max(average_return4),
    # max(average_return5), max(average_return6), max(average_return7), max(average_return8)]

    NN = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}
    plt.figure(figsize=(10, 8))
    plt.plot(NN, data_1, marker='o', label='BatteryLevel1', markersize=10)
    plt.plot(NN, data_2, marker='*', label='BatteryLevel2', markersize=10)
    plt.plot(NN, data_3, marker='x', label='BatteryLevel3', markersize=10)
    plt.legend(loc=4)
    plt.legend(prop=font1, edgecolor='black', facecolor='white')
    plt.tick_params(labelsize=20)
    plt.ylabel('Average throughput', font2)
    plt.xlabel('Size of Memory', font2)
    plt.savefig('super1.eps', bbox_inches='tight')
    plt.show()

    """
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}
    # 平均吞吐量线性表示
    plt.figure(num=1, figsize=(10, 8))
    # plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    plt.plot(np.linspace(0, len(average_return1), N), average_return1, label='BatteryLevel1', markersize=10)
    plt.plot(np.linspace(0, len(average_return2), N), average_return2, label='BatteryLevel2', markersize=10)
    plt.plot(np.linspace(0, len(average_return3), N), average_return3, label='BatteryLevel3', markersize=10)
    plt.legend(loc=4)
    plt.legend(prop=font1, edgecolor='black', facecolor='white')
    plt.tick_params(labelsize=20)
    plt.ylabel('Average throughput', font2)
    plt.xlabel('Training episodes', font2)
    plt.savefig('average_return1.eps', bbox_inches='tight')
    plt.show()

    # 平均吞吐量点线表示
    plt.figure(num=2, figsize=(10, 8))
    x1 = []
    x2 = []
    x3 = []
    g1 = []
    g2 = []
    g3 = []
    for i in range(len(average_return1)):
        if i % 40 == 0:
            x1.append(np.arange(len(average_return1))[i])
            g1.append(average_return1[i])
    for i in range(len(average_return2)):
        if i % 40 == 0:
            x2.append(np.arange(len(average_return2))[i])
            g2.append(average_return2[i])
    for i in range(len(average_return2)):
        if i % 40 == 0:
            x3.append(np.arange(len(average_return3))[i])
            g3.append(average_return3[i])
    plt.plot(x1, g1, marker='o', label='BatteryLevel1', markersize=10)
    plt.plot(x2, g2, marker='x', label='BatteryLevel2', markersize=10)
    plt.plot(x3, g3, marker='^', label='BatteryLevel3', markersize=10)
    plt.legend(loc=4)
    plt.legend(prop=font1, edgecolor='black', facecolor='white')
    plt.tick_params(labelsize=20)
    plt.ylabel('Average throughput', font2)
    plt.xlabel('Training episodes', font2)
    plt.savefig('average_return2.eps', bbox_inches='tight')
    plt.show()

    # 回合吞吐量线性表示
    plt.figure(num=3, figsize=(10, 8))
    plt.plot(np.arange(len(episode_return1)), episode_return1)
    plt.plot(np.arange(len(episode_return2)), episode_return2)
    plt.plot(np.arange(len(episode_return3)), episode_return3)
    plt.ylabel('Episode_Return')
    plt.xlabel('Training episodes')
    plt.show()

    # 回合吞吐量点线表示
    plt.figure(num=4, figsize=(10, 8))
    x4 = []
    y4 = []
    for i in range(len(episode_return1)):
        if i % 10 == 0:
            x4.append(np.arange(len(episode_return1))[i])
            y4.append(episode_return1[i])
    plt.plot(x4, y4, marker='o')
    plt.show()
    """

    env.reset_uav()
    env.render()
    end = time.time()
    print("game over!")
    print('运行时间:', end - start)
    engine = pyttsx3.init()
    engine.say('程序运行完成')
    engine.runAndWait()
    env.destory()


if __name__ == "__main__":
    env = Maze()
    RL1 = DeepQNetwork1(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=200,  # 尝试减少替换次数
                        memory_size=2000,  # 尝试扩大记忆库
                        output_graph=False
                        )

    RL2 = DeepQNetwork2(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=180,  # 尝试减少替换次数
                        memory_size=1800,  # 尝试扩大记忆库
                        )

    RL3 = DeepQNetwork3(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=160,  # 尝试减少替换次数
                        memory_size=1600,  # 尝试扩大记忆库
                        )

    RL4 = DeepQNetwork4(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=140,  # 尝试减少替换次数
                        memory_size=1400,  # 尝试扩大记忆库
                        )

    RL5 = DeepQNetwork5(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=120,  # 尝试减少替换次数
                        memory_size=1200,  # 尝试扩大记忆库
                        )

    RL6 = DeepQNetwork6(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=100,  # 尝试减少替换次数
                        memory_size=1000,  # 尝试扩大记忆库
                        )

    RL7 = DeepQNetwork7(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=80,  # 尝试减少替换次数
                        memory_size=800,  # 尝试扩大记忆库
                        )

    RL8 = DeepQNetwork8(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=60,  # 尝试减少替换次数
                        memory_size=600,  # 尝试扩大记忆库
                        )

    env.after(100, run_maze)
    env.mainloop()

    """
    plt.figure(figsize=(10, 8))
    RL.plot_cost()
    RL_.plot_cost()
    RL__.plot_cost()
    plt.savefig('Loss1.eps')
    plt.show()
    """