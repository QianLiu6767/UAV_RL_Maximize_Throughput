import numpy as np
import matplotlib.pyplot as plt


theta = np.array([1, 2, 2, 3])
gamma = np.array([2, 10, 5, 0.6])
la = np.array(['θ=1, γ=2', 'θ=2, γ=10', 'θ=2, γ=5', 'θ=3, γ=0.6'])
ma = np.array(['o', 'v', 'x', 'p'])
x = np.arange(0, 10, 1)

plt.figure(figsize=(10, 8))

for i in range(len(theta)):
    u = 1 - np.exp(-((x ** theta[i]) / (x + gamma[i])))
    plt.plot(x, u, label=la[i], marker=ma[i], markersize=10)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20}

plt.legend(prop=font1, edgecolor='black', facecolor='white')

plt.tick_params(labelsize=20)

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 22}
plt.xlabel("The number of offloading tasks", font2)
plt.ylabel("System gain", font2)

plt.savefig('systemgain.eps')
plt.show()



"""
plt.figure(figsize=(10, 8))
for i in range(len(theta)):
    u = 1 - np.exp(-((x ** theta[i]) / (x + gamma[i])))
    plt.plot(x, u, label=la[i], marker=ma[i])


plt.legend()

plt.xlabel("The number of offloading tasks")
plt.ylabel("System gain")

"""

#plt.show()
