"""
Task 01 Polynomial Regression
"""

#%%

# Import the libraries used:
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def feature_expansion(features, degree):
    X = np.ones(features.shape)

    for i in range(1, degree + 1):
        X = np.column_stack((X, features ** i))

    return X

def pol_regression(x, y, degree):
    X = feature_expansion(x, degree=degree)

    # Least Square:
    first_half = X.transpose().dot(X)
    weights = np.linalg.solve(first_half, X.transpose().dot(y))

    return weights

def calculate_y(x, weights):
    total = 0
    total += weights[0]

    for i, weight in enumerate(weights):
        if i == 0:
            total += weight * (x ** i)
        else:
            val = weight * (x ** i)
            total += val

    return total

def plot_predictions(weights, features):
    y =[]
    for x in features:
        y.append(calculate_y(x, weights))

    return y

# Get data:
data_frame = pd.read_csv('polyRegression.csv')
x_train = data_frame['x']
y_train = data_frame['y']

# Calculate the weights:
weights_degree1 = pol_regression(x_train, y_train, degree=1)
weights_degree2 = pol_regression(x_train, y_train, degree=2)
weights_degree3 = pol_regression(x_train, y_train, degree=3)
weights_degree6 = pol_regression(x_train, y_train, degree=6)
weights_degree10 = pol_regression(x_train, y_train, degree=10)

# Calculate the Y(hat) predicted:
y_hat1 = plot_predictions(weights=weights_degree1, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y_hat2 = plot_predictions(weights=weights_degree2, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y_hat3 = plot_predictions(weights=weights_degree3, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y_hat6 = plot_predictions(weights=weights_degree6, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y_hat10 = plot_predictions(weights=weights_degree10, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

# Display the graphs:
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat1, color='red')
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat2, color='green')
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat3, color='purple')
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat6, color='yellow')
plt.scatter(x_train, y_train)
plt.show()

# %%
