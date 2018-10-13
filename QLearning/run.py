# Q-learning 算法解决UAV

from Maze import Maze
from Brains import Dream
import matplotlib.pyplot as plt
import numpy as np


def update():
    remark = []
    remark_ = []
    migration = []
    average_migration = []

    cumulative_reward = 0
    g = 0  # Average return rate

    U = 0
    t = 0  # updates times

    global q_table

    for episode in range(1000):

        observation = env.reset()

        while True:

            env.render()

            action = RL.choose_action(str(observation))

            observation_, reward,  done, u = env.step(action)         # tau为step计数器

            q_table = RL.learn(str(observation), action, reward, str(observation_))

            t += 1                              # 更新次数加一

            U += u                              # 累计吞吐量

            cumulative_reward += reward         # 累计奖励

            if done:
                g = cumulative_reward / t  # 平均奖励率
                remark_.append(cumulative_reward)  # 记录累计奖励
                remark.append(g)  # 记录平均奖励率
                migration.append(U)
                average_migration.append(U / t)  # 记录平均吞吐量
                break

            observation = observation_

    # 平均奖励
    plt.figure(figsize=(10, 8))

    x1 = []
    g1 = []
    for i in range(len(remark)):
        if i % 50 == 0:
            x1.append(np.arange(len(remark))[i])
            g1.append(remark[i])

    plt.plot(x1, g1, marker='o', label='BatteryLevel1', markersize=10)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}

    plt.legend(loc=4)

    plt.legend(prop=font1, edgecolor='black', facecolor='white')

    plt.tick_params(labelsize=20)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 22}

    plt.ylabel('Average Return', font2)
    plt.xlabel('Training Episodes', font2)
    plt.savefig('average_return.eps', bbox_inches='tight')
    plt.show()

    # 平均吞吐量
    plt.figure(figsize=(10, 8))
    x2 = []
    g2 = []

    for i in range(len(average_migration)):
        if i % 50 == 0:
            x2.append(np.arange(len(average_migration))[i])
            g2.append(average_migration[i])

    plt.plot(x2, g2, marker='x', label='BatteryLevel2', markersize=10)

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
    plt.plot(np.arange(len(remark)), remark)
    plt.xlabel("Total Episodes")
    plt.ylabel("Average Return")
    plt.figure(2)
    plt.plot(np.arange(len(average_migration)), average_migration)
    plt.xlabel("Total Episodes")
    plt.ylabel("Average Throughput")
    plt.show()
    """

    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = Dream(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
