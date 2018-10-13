from Maze import Maze
from RL_brain import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3

start = time.time()

def run_maze():
    cumulative_reward = 0
    remark = []
    origin = []
    origin_ = 0
    average_return = 0
    step = 0

    for episode in range(1000):

        observation = env.reset_uav()

        env.render()
        # time.sleep(1)

        while True:

            env.render()

            action, flag = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            if flag:
                cumulative_reward += reward
                origin_ += reward
                average_return = cumulative_reward / step
            else:
                pass

            RL.store_transition(observation, action, reward, observation_, average_return)

            if(step > 100) and done:
                RL.learn()

            observation = observation_

            if done:
                remark.append(average_return)
                origin.append(origin_)
                origin_ = 0
                break
            step += 1

    plt.figure(1)
    plt.plot(np.arange(len(remark)), remark)
    plt.ylabel('Average_Return')
    plt.xlabel('training episodes')

    plt.figure(2)
    x = []
    y = []
    for i in range(len(remark)):
        if i % 20 == 0:
            x.append(np.arange(len(remark))[i])
            y.append(remark[i])
    plt.plot(x, y, marker='o')
    plt.legend()

    plt.figure(3)
    plt.plot(np.arange(len(origin)), origin)
    plt.ylabel('Episode_Return')
    plt.xlabel('training episodes')
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
                      learning_rate=0.02,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 尝试减少替换次数100
                      memory_size=2000,  # 尝试扩大记忆库6000
                      output_graph=False
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()  # 观察神经网络的误差曲线
