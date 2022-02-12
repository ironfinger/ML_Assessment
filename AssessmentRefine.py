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

def eval_pol_regression(parameters, x, y, degree):
    # X = actual feature.
    # Y = y_hat
    
    # First set is to compute the mean squared error:
    MSE = np.square(np.subtract(x, y)).mean()
    RMSE = np.sqrt(MSE)
    return RMSE
    

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
from sklearn.model_selection import train_test_split
# Compute errors:
# Step 01: Split the training data:
train_df, test_df = train_test_split(data_frame, test_size=0.3)

# Get the independant and dependant from the test and train data frames:
x_train = train_df['x']
y_train = train_df['y']
x_test = test_df['x']
y_test = test_df['y']

# Step 02: Train the models with the test set with the same degrees:
weights_degree1 = pol_regression(x_train, y_train, degree=1)
weights_degree2 = pol_regression(x_train, y_train, degree=2)
weights_degree3 = pol_regression(x_train, y_train, degree=3)
weights_degree6 = pol_regression(x_train, y_train, degree=6)
weights_degree10 = pol_regression(x_train, y_train, degree=10)

# Calculate the y_hats:
y_hat1 = plot_predictions(weights=weights_degree1, features=x_train)
y_hat2 = plot_predictions(weights=weights_degree2, features=x_train)
y_hat3 = plot_predictions(weights=weights_degree3, features=x_train)
y_hat6 = plot_predictions(weights=weights_degree6, features=x_train)
y_hat10 = plot_predictions(weights=weights_degree10, features=x_train)

# Lets see the data before we do anything:
print(len(x_train))
print(len(y_train))
print(len(y_hat1))

# Compute the RMSE on the training set:
RMSE_degree1 = eval_pol_regression(parameters=weights_degree1, x=y_train, y=y_hat1, degree=1)
RMSE_degree2 = eval_pol_regression(parameters=weights_degree2, x=y_train, y=y_hat2, degree=2)
RMSE_degree3 = eval_pol_regression(parameters=weights_degree3, x=y_train, y=y_hat3, degree=3)
RMSE_degree6 = eval_pol_regression(parameters=weights_degree6, x=y_train, y=y_hat6, degree=6)
RMSE_degree10 = eval_pol_regression(parameters=weights_degree10, x=y_train, y=y_hat10, degree=10)

errors_train = [RMSE_degree1, RMSE_degree2, RMSE_degree3, RMSE_degree6, RMSE_degree10]
degrees = [1, 2, 3, 6, 10]

plt.plot(degrees, errors_train, color='red')
plt.show()

#%%

y_hat1 = plot_predictions(weights=weights_degree1, features=x_test)
y_hat2 = plot_predictions(weights=weights_degree2, features=x_test)
y_hat3 = plot_predictions(weights=weights_degree3, features=x_test)
y_hat6 = plot_predictions(weights=weights_degree6, features=x_test)
y_hat10 = plot_predictions(weights=weights_degree10, features=x_test)

# Compute the RMSE on the test set:
RMSE_degree1 = eval_pol_regression(parameters=weights_degree1, x=y_test, y=y_hat1, degree=1)
RMSE_degree2 = eval_pol_regression(parameters=weights_degree2, x=y_test, y=y_hat2, degree=2)
RMSE_degree3 = eval_pol_regression(parameters=weights_degree3, x=y_test, y=y_hat3, degree=3)
RMSE_degree6 = eval_pol_regression(parameters=weights_degree6, x=y_test, y=y_hat6, degree=6)
RMSE_degree10 = eval_pol_regression(parameters=weights_degree10, x=y_test, y=y_hat10, degree=10)

errors_test = [RMSE_degree1, RMSE_degree2, RMSE_degree3, RMSE_degree6, RMSE_degree10]
degrees = [1, 2, 3, 6, 10]
           
           
plt.plot(degrees, errors_test, color='red')
plt.show()

# %%
