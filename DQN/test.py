import numpy as np
import matplotlib.pyplot as plt

a = [1, 2, 4, 7, 11, 15, 20, 26, 33, 41]
b = []

x1 = np.arange(len(a))
x2 = []
for i in range(len(a)):
    if i % 2 == 0:
        x2.append(np.arange(len(a))[i])

for i in range(len(a)):
    if i % 2 == 0:
        b.append(a[i])
plt.plot(x1, a, marker='o', label='a')
plt.plot(x2, b, marker='^', label='b')
plt.legend()
plt.show()
