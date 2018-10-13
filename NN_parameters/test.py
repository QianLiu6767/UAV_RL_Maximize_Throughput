import numpy as np
import matplotlib.pyplot as plt

a = [3.1, 5.9, 9.6]
b = [4, 4, 4]
c = [3, 3, 3]
x = [1, 2, 3]
fig = plt.figure(figsize=(10, 8))
plt.plot(x, a, marker='o')
plt.plot(x, b, marker='*')
plt.plot(x, c, marker='x')
print(sum(b)/len(b))
