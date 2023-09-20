import numpy
import torch
import matplotlib.pyplot as plt

x = torch.randn(12).reshape(3, 4)
y = torch.randn(12).reshape(3, 4)
x = torch.randn(12)
y = torch.randn(12)

fig1 = plt.figure
plt.scatter(x, y)
plt.show()

fig2 = plt.figure
plt.plot(x, y, 'r-')
plt.show()
print(x, y)