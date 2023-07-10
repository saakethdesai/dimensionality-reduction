import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("train.out", skiprows=5, usecols=(9, 10, 11))
print (data)

plt.plot(data[:, 0], 'o-', color='blue', label='train')
plt.plot(data[:, 1], 'o-', color='orange', label='validation')
plt.plot(data[:, 2], 'o-', color='green', label='test')
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.savefig("error.png")
