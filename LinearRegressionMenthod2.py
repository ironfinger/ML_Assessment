"""
PLEASE PLEASE PLEASE
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%

dataset = pd.read_csv('https://raw.githubusercontent.com/FutrCamp/Machine-Learning/main/Supervised%20Machine%20Learning/Datasets/data_1d.csv', header=None)
dataset.head()
# %%

X = dataset[0]
Y = dataset[1]

print(len(X))
print(len(Y))

# %%

# Mean Squared Error:
def MSE(m, b, X, Y):
    total_error = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(X))

# link: https://www.youtube.com/watch?v=VmbA0pi2cRQ
def gradient_desc(m_now, b_now, X, Y, L):
    m_gradient = 0
    b_gradient = 0

    n = len(X)

    for i in range(n):
        x = X[i]
        y = Y[i]

        # This is the partial derivative of MSE in respect to m
        m_gradient += (-2/n) * x * (y - (m_now * x + b_now))

        # This is the partial derivative of MSE in respsect to b (the intercept)
        b_gradient += (-2/n) * x * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = m_now - b_gradient * L

    return m, b

m = 0
b = 0
L = 0.000001
epochs = 1000

for i in range(epochs):
    m, b = gradient_desc(m, b, X, Y, L)

    if (i % 50) == 0:
        print('epoch: ', i)
        plt.scatter(X, Y, color='black')
        plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color='red')
        plt.show()

print(m, b)

plt.scatter(X, Y, color='black')
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color='red')
plt.show()
# %%

"""
Attempt 2:
"""