import numpy as np
import matplotlib.pyplot as plt

theta = np.array([1, 2, 2, 3])
gamma = np.array([2, 10, 5, 0.6])
la = np.array(['θ=1, γ=2', 'θ=2, γ=10', 'θ=2, γ=5', 'θ=3, γ=0.6'])
ma = np.array(['o', 'v', 'x', 'p'])
x = np.arange(0, 10, 1)
for i in range(len(theta)):
    u = 1 - np.exp(-((x ** theta[i]) / (x + gamma[i])))
    plt.plot(x, u, label=la[i], marker=ma[i])
plt.legend()
plt.xlabel("data size")
plt.ylabel("System income")
plt.show()

