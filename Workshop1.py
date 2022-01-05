"""
Linear Regression Tutorial: https://www.youtube.com/watch?v=aI0KqA5Q34E&list=PLYmq7TpgsPl2gPKDqW8LqFwadQVLe5Mgg&index=2
Polynomial Regression Tutorial: https://www.youtube.com/watch?v=5o18M_muUZE
"""

#%% 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%

dataset = pd.read_csv('https://raw.githubusercontent.com/FutrCamp/Machine-Learning/main/Supervised%20Machine%20Learning/Datasets/data_1d.csv', header=None)
dataset.head()

# %%

# Split the data into independent and dependant variable.
# X = independent variable
# Y = dependent variable
X = np.array(dataset[0])
Y = np.array(dataset[1])

print(X.shape)
# %%

# Visualise the data:
plt.scatter(X, Y)
plt.show()

# %%

# Calculate the weights:
denominator = X.dot(X) - X.mean()*X.sum()
a = (X.dot(Y) - Y.mean()*X.sum()) / denominator
b = (Y.mean()*X.dot(X) - X.mean()*X.dot(Y)) / denominator

# %%

# Calculate predicted Y value:
Yhat = a*X + b
print(Yhat)
# %%

# Show the linear regression:
plt.scatter(X, Y)
plt.plot(X, Yhat, color='red')
# %%

a = np.array([2, 2, 2])
b = np.array([3, 4, 5])

print(a.dot(b))

# %%
