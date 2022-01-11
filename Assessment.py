"""
Task 01:
- Solve a polynomial regression example.
- You are requied to analyse the performance of a polynomial regression algorithm.
    - This is through fitting it to the data and estimating the degree of the polynomial,
        as well as its parameters.

- Implement the algorithm.
- Evaluate its performance on the training as well on an independent test set.
"""

#%%

"""
Attempt 01
"""

# Get the data:
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Get the data as a pandas dataframe:
df = pd.read_csv(os.path.join('regression_train.csv'))

def feature_expansion(features, degree):
    X = np.ones(features.shape)

    for i in range(1, degree + 1):
        print('Degree: ', i)
        
        X = np.column_stack((X, features ** i))

    return X

# print('Without libraries !!!!!')
# print(feature_expansion(df['x'], 2))

# print('With Libraries')
# transform = PolynomialFeatures(degree=2)
# x = np.array(df['x'])
# x = x[:, np.newaxis]
# data = transform.fit_transform(x)
# print(data)

def get_weights(x, y, degree):
    X = feature_expansion(x, degree)

    # Least Square:
    first_half = X.transpose().dot(X)
    weights = np.linalg.solve(first_half, X.transpose().dot(y))

    return weights

# print('Without Libraries')
# print(get_weights(df['x'], df['y'], degree=2))
# print('With')
# from sklearn.linear_model import LinearRegression
# transform = PolynomialFeatures(degree=2)
# x = np.array(df['x'])
# x = x[:, np.newaxis]
# expansion = transform.fit_transform(x)
# poly = LinearRegression()
# poly.fit(expansion, df['y'])
# print(poly.coef_)

plt.show()

# %%
"""
Attempt 02:
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def feature_expansion(features, degree):
    X = np.ones(features.shape)

    for i in range(1, degree + 1):
        print('Degree: ', i)
        
        X = np.column_stack((X, features ** i))

    return X

def get_weights(x, y, degree):
    X = feature_expansion(x, degree)

    # Least Square:
    first_half = X.transpose().dot(X)
    weights = np.linalg.solve(first_half, X.transpose().dot(y))

    return weights

def pol_regression(features_train, y_train, degree):
    return get_weights(features_train, y_train, degree)

# Get data:
data_frame = pd.read_csv('polyRegression.csv')
data_frame.head()

x_train = data_frame['x']
y_train = data_frame['y']

def calculate_y(x, weights):
    
    total = 0
    total += weights[0]

    for i, weight in enumerate(weights):
        if i == 0:
            total += weight
        else:
            val = weight * (x ** i)
            total += val

    return total

def plot_predictions(weights, features):
    y = []
    for x in features:
        y.append(calculate_y(x, weights))

    return y
        
# Calculate the weights:
weights_degree1 = pol_regression(x_train, y_train, degree=1)
weights_degree2 = pol_regression(x_train, y_train, degree=2)
weights_degree3 = pol_regression(x_train, y_train, degree=3)
weights_degree6 = pol_regression(x_train, y_train, degree=6)
weights_degree10 = pol_regression(x_train, y_train, degree=10)

# Calculate the Y predicted:
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

# Split the data set:
# Split the training data
from sklearn.model_selection import train_test_split

x = data_frame['x']
y = data_frame['y']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

plt.scatter(X_test, Y_test, color='red')
plt.show()
# %%


y_hat_test_degree1 = plot_predictions(weights=weights_degree1, features=X_test)
y_hat_test_degree2 = plot_predictions(weights=weights_degree2, features=X_test)
y_hat_test_degree3 = plot_predictions(weights=weights_degree3, features=X_test)
y_hat_test_degree6 = plot_predictions(weights=weights_degree6, features=X_test)
y_hat_test_degree10 = plot_predictions(weights=weights_degree10, features=X_test)

plt.scatter(X_test, y_hat_test_degree1, color='red')
plt.scatter(X_test, y_hat_test_degree2, color='blue')
plt.scatter(X_test, y_hat_test_degree3, color='yellow')
plt.scatter(X_test, y_hat_test_degree6, color='purple')
plt.scatter(X_test, y_hat_test_degree10, color='black')
plt.show()

# %%
import math

# Calculate RMSE:
def RMSE(y_actual, y_hat):
    """Calculate the RMSE

    Args:
        y_actual ([float]): [the y values in the test set]
        y_hat ([float]): [the y values predeicted by the model]
    """

    Mean_squared_error = np.square(np.subtract(y_actual, y_hat)).mean()
    Root_mean_squared = math.sqrt(Mean_squared_error)
    return Root_mean_squared

print('Here are the Root mean squared error')
print('Degree 1: ', 'RME=', RMSE(Y_test, y_hat_test_degree1))
print('Degree 2: ', 'RME=', RMSE(Y_test, y_hat_test_degree2))
print('Degree 3: ', 'RME=', RMSE(Y_test, y_hat_test_degree3))
print('Degree 6: ', 'RME=', RMSE(Y_test, y_hat_test_degree6))
print('Degree10: ', 'RME=', RMSE(Y_test, y_hat_test_degree10))
print('whoop whoop')
# %%
