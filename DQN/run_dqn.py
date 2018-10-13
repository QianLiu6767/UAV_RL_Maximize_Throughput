from Maze import Maze
from DQN_brain import DeepQNetwork
from DQN_brain import DeepQNetwork2
from DQN_brain import DeepQNetwork3
import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3


start = time.time()

#  运行前是否调整epsilon


def run_maze():
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    step = 0

    for episode in range(1000):

        observation = env.reset_uav()
        env.render()
        #time.sleep(1)

        while True:

            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            cumulative_reward1 += reward
            cumulative_reward1_ += reward

            RL.store_transition(observation, action, reward, observation_)

            if(step > 400) and done:
                RL.learn()

            observation = observation_

            step += 1

            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                cumulative_reward1_ = 0
                break

    cumulative_reward2 = 0
    cumulative_reward2_ = 0
    average_return2 = []
    episode_return2 = []
    step = 0

    for episode in range(1000):

        observation = env.reset_uav_()
        env.render()
        #time.sleep(1)

        while True:

            env.render()

            action = RL_.choose_action(observation)

            observation_, reward, done = env.step(action)

            cumulative_reward2 += reward
            cumulative_reward2_ += reward

            RL_.store_transition_(observation, action, reward, observation_)

            if(step > 400) and done:
                RL_.learn()

            observation = observation_

            step += 1

            if done:
                average_return2.append(cumulative_reward2 / step)
                episode_return2.append(cumulative_reward2_)
                cumulative_reward2_ = 0
                break

    cumulative_reward3 = 0
    cumulative_reward3_ = 0
    average_return3 = []
    episode_return3 = []
    step = 0

    for episode in range(1000):

        observation = env.reset_uav__()
        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action = RL__.choose_action(observation)

            observation_, reward, done = env.step(action)

            cumulative_reward3 += reward
            cumulative_reward3_ += reward

            RL__.store_transition__(observation, action, reward, observation_)

            if (step > 400) and done:
                RL__.learn()

            observation = observation_

            step += 1

            if done:
                average_return3.append(cumulative_reward3 / step)
                episode_return3.append(cumulative_reward3_)
                cumulative_reward3_ = 0
                break
    plt.figure(1)
    #plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    plt.plot(np.linspace(0, len(average_return1), 1000), average_return1, label='BatteryLevel1')
    plt.plot(np.linspace(0, len(average_return2), 1000), average_return2, label='BatteryLevel2')
    plt.plot(np.linspace(0, len(average_return3), 1000), average_return3, label='BatteryLevel3')
    plt.legend(loc=4)
    plt.ylabel('Average_Return')
    plt.xlabel('training episodes')
    plt.show()

    plt.figure(2)
    x1 = []
    x2 = []
    x3 = []
    g1 = []
    g2 = []
    g3 = []
    for i in range(len(average_return1)):
        if i % 60 == 0:
            x1.append(np.arange(len(average_return1))[i])
            g1.append(average_return1[i])
    for i in range(len(average_return2)):
        if i % 60 == 0:
            x2.append(np.arange(len(average_return2))[i])
            g2.append(average_return2[i])
    for i in range(len(average_return2)):
        if i % 60 == 0:
            x3.append(np.arange(len(average_return3))[i])
            g3.append(average_return3[i])
    plt.plot(x1, g1, marker='o', label='BatteryLevel1')
    plt.plot(x2, g2, marker='x', label='BatteryLevel2')
    plt.plot(x3, g3, marker='^', label='BatteryLevel3')
    plt.legend(loc=4)
    plt.ylabel('Average_Return')
    plt.xlabel('training episodes')
    plt.show()

    plt.figure(3)
    plt.plot(np.arange(len(episode_return1)), episode_return1)
    plt.plot(np.arange(len(episode_return2)), episode_return2)
    plt.plot(np.arange(len(episode_return3)), episode_return3)
    plt.ylabel('Episode_Return')
    plt.xlabel('training episodes')
    plt.show()

    plt.figure(4)
    x4 = []
    y4 = []
    for i in range(len(episode_return1)):
        if i % 10 == 0:
            x4.append(np.arange(len(episode_return1))[i])
            y4.append(episode_return1[i])
    plt.plot(x4, y4, marker='o')
    plt.show()
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
    RL.plot_cost()
    RL_.plot_cost()
    RL__.plot_cost()
