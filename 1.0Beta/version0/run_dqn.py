from Maze import Maze
from DQN_brain import DeepQNetwork
from DQN_brain import DeepQNetwork2
from DQN_brain import DeepQNetwork3
import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3

N = 1000
start = time.time()

#  运行前是否调整epsilon


def run_maze():
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []

    migration1 = []
    average_migration1 = []

    m1 = 0
    step = 0

    for episode in range(N):

        observation = env.reset_uav()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward
            m1 += u                     # 记录每一步的迁移量

            RL.store_transition(observation, action, reward, observation_)

            if (step > 400) and done:
                RL.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                migration1.append(m1)
                average_migration1.append(m1 / step)
                cumulative_reward1_ = 0
                break

    cumulative_reward2 = 0
    cumulative_reward2_ = 0
    average_return2 = []
    episode_return2 = []
    m2 = 0
    step = 0
    migration2 = []
    average_migration2 = []

    for episode in range(N):

        observation = env.reset_uav_()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL_.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward2 += reward
            cumulative_reward2_ += reward
            m2 += u  # 记录每一步的迁移量

            RL_.store_transition_(observation, action, reward, observation_)

            if(step > 400) and done:
                RL_.learn()

            observation = observation_

            step += 1

            if done:
                average_return2.append(cumulative_reward2 / step)
                episode_return2.append(cumulative_reward2_)
                migration2.append(m2)
                average_migration2.append(m2 / step)
                cumulative_reward2_ = 0
                break

    cumulative_reward3 = 0
    cumulative_reward3_ = 0
    average_return3 = []
    episode_return3 = []
    m3 = 0
    step = 0
    migration3 = []
    average_migration3 = []

    for episode in range(N):

        observation = env.reset_uav__()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL__.choose_action(observation)

            observation_, reward, done, u = env.step(action)

            cumulative_reward3 += reward
            cumulative_reward3_ += reward
            m3 += u  # 记录每一步的迁移量

            RL__.store_transition__(observation, action, reward, observation_)

            if (step > 400) and done:
                RL__.learn()

            observation = observation_

            step += 1

            if done:
                average_return3.append(cumulative_reward3 / step)
                episode_return3.append(cumulative_reward3_)
                migration3.append(m3)
                average_migration3.append(m3 / step)
                cumulative_reward3_ = 0
                break

    # 相同学习率，不同电量的平均吞吐量对比图

    plt.figure(figsize=(10, 8))
    # plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    x1 = []
    x2 = []
    x3 = []
    g1 = []
    g2 = []
    g3 = []
    for i in range(len(average_migration1)):
        if i % 50 == 0:
            x1.append(np.arange(len(average_migration1))[i])
            g1.append(average_migration1[i])

    for i in range(len(average_migration2)):
        if i % 50 == 0:
            x2.append(np.arange(len(average_migration2))[i])
            g2.append(average_migration2[i])
    for i in range(len(average_migration3)):
        if i % 50 == 0:
            x3.append(np.arange(len(average_migration3))[i])
            g3.append(average_migration3[i])

    plt.plot(x1, g1, marker='o', label='BatteryLevel1', markersize=10)
    plt.plot(x2, g2, marker='x', label='BatteryLevel2', markersize=10)
    plt.plot(x3, g3, marker='^', label='BatteryLevel3', markersize=10)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    plt.legend(loc=4)

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}

    plt.ylabel('Average Throughput', font2)
    plt.xlabel('Training Episodes', font2)
    plt.savefig('average_migration.eps', bbox_inches='tight')
    plt.show()

    """
    plt.figure(1)
    plt.plot(np.arange(len(migration1)), migration1)
    plt.xlabel("Total Episodes")
    plt.ylabel("Total Migration")
    plt.figure(2)
    plt.plot(np.arange(len(average_migration1)), average_migration1)
    plt.xlabel("Total Episodes")
    plt.ylabel("Average Migration")
    plt.show()
    """
    
    # 相同学习率，不同电量的平均奖励对比图
    plt.figure(figsize=(10, 8))
    x1 = []
    x2 = []
    x3 = []
    r1 = []
    r2 = []
    r3 = []
    for i in range(len(average_return1)):
        if i % 50 == 0:
            x1.append(np.arange(len(average_return1))[i])
            r1.append(average_return1[i])
    for i in range(len(average_return2)):
        if i % 50 == 0:
            x2.append(np.arange(len(average_return2))[i])
            r2.append(average_return2[i])
    for i in range(len(average_return2)):
        if i % 50 == 0:
            x3.append(np.arange(len(average_return3))[i])
            r3.append(average_return3[i])
    plt.plot(x1, r1, marker='o', label='BatteryLevel1', markersize=10)
    plt.plot(x2, r2, marker='x', label='BatteryLevel2', markersize=10)
    plt.plot(x3, r3, marker='^', label='BatteryLevel3', markersize=10)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    plt.legend(loc="lower right")

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}

    plt.ylabel('Average Return', font2)
    plt.xlabel('Training Episodes', font2)
    plt.savefig('average_return.eps', bbox_inches='tight')
    plt.show()

    """
    # 回合奖励
    # plt.figure(3)
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(len(episode_return1)), episode_return1, markersize=10)
    plt.plot(np.arange(len(episode_return2)), episode_return2, markersize=10)
    plt.plot(np.arange(len(episode_return3)), episode_return3, markersize=10)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    # plt.legend(loc=4)

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}

    plt.ylabel('Episode_Return'), font2
    plt.xlabel('Training episodes', font2)
    plt.show()

    
    # 回合奖励的采样图
    plt.figure(4)
    x4 = []
    y4 = []
    for i in range(len(episode_return1)):
        if i % 50 == 0:
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
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 尝试减少替换次数
                      memory_size=2000,  # 尝试扩大记忆库
                      output_graph=False
                      )

    RL_ = DeepQNetwork2(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=200,  # 尝试减少替换次数
                        memory_size=2000,  # 尝试扩大记忆库
                        )
    RL__ = DeepQNetwork3(env.n_actions, env.n_features,
                         learning_rate=0.01,
                         reward_decay=0.9,
                         e_greedy=0.9,
                         replace_target_iter=200,  # 尝试减少替换次数
                         memory_size=2000,  # 尝试扩大记忆库
                         )

    env.after(100, run_maze)
    env.mainloop()

    """
    # 不同学习率的cost图
    plt.figure(figsize=(10, 8))
    RL.plot_cost()
    RL_.plot_cost()
    RL__.plot_cost()
    plt.savefig('Loss.eps')
    plt.show()
    """
