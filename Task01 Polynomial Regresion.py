"""
Task 01 Polynomial Regression
"""

#%%

# Import the libraries used:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def feature_expansion(features, degree):
    
    # Create a matrix full of ones with the same shape as the features:
    X = np.ones(features.shape)

    # Generate the feature matrix of all combinations of the polynomial:
    for i in range(1, degree + 1):
        X = np.column_stack((X, features ** i))

    return X

def pol_regression(x, y, degree):
    
    # Call the feature expansion function to expand the polynomial features:
    X = feature_expansion(x, degree=degree)

    # Least Square method:
    
    # Calculates: X(transpose) dot/times X
    first_half = X.transpose().dot(X) # calculate the first half of the equation.
    
    # Calculate the second half thus returning the weights.
    # Calculates the inverse of the first half dot/times X(transpose) dot/times y
    weights = np.linalg.solve(first_half, X.transpose().dot(y))
    
    return weights

def calculate_y(x, weights, degree):
    total = 0
    
    # Loop through the weights to gather the degree of the polynomial.
    for i, weight in enumerate(weights):
        
        # calculate the current degree of the polynomial
        val = weight * (x ** i)
        total += val

    return total

def plot_predictions(weights, features, degree):
    y =[]
    
    # Loop through each feaure to calculate the predicted y value:
    for x in features:
        y.append(calculate_y(x, weights, degree))

    return y

def eval_pol_regression(parameters, x, y, degree):
    
    # X is the x_test
    # y is the y_test
    
    # Compute the predicted y values.
    y_hat = plot_predictions(weights=parameters, features=x, degree=degree)
    
    # Firstly we must compute the mean squared error:
    MSE = np.square(np.subtract(y_hat, y)).mean()
    
    # Apply the square root to retreive the RMSE:
    RMSE = np.sqrt(MSE)
    return RMSE


# Get data:
# data_frame = pd.read_csv('Task1 - dataset - pol_regression.csv')
data_frame = pd.read_csv('polyRegression.csv')
x_train = data_frame['x']
y_train = data_frame['y']

# Calculate the weights:
#print('Degree 0') # INCLUDE THIS IN THE REPORT !!!
#weights_degree0 = pol_regression(x_train, y_train, degree=0)

weights_degree0 = pol_regression(x_train, y_train, degree=0)
weights_degree1 = pol_regression(x_train, y_train, degree=1)
weights_degree2 = pol_regression(x_train, y_train, degree=2)
weights_degree3 = pol_regression(x_train, y_train, degree=3)
weights_degree6 = pol_regression(x_train, y_train, degree=6)
weights_degree10 = pol_regression(x_train, y_train, degree=10)

# Calculate the Y(hat) predicted:

y_hat0 = plot_predictions(weights=weights_degree1, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], degree=0)
y_hat1 = plot_predictions(weights=weights_degree1, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], degree=1)
y_hat2 = plot_predictions(weights=weights_degree2, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], degree=2)
y_hat3 = plot_predictions(weights=weights_degree3, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], degree=3)
y_hat6 = plot_predictions(weights=weights_degree6, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], degree=6)
y_hat10 = plot_predictions(weights=weights_degree10, features=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], degree=10)

# Display the graphs:
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat1, color='red', label='Degree = 0')
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat1, color='red', label='Degree = 1')
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat2, color='green', label='Degree = 2')
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat3, color='purple', label='Degree = 3')
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat6, color='yellow', label='Degree = 6')
plt.plot([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], y_hat10, color='black', label='Degree = 10')
plt.scatter(x_train, y_train)
plt.legend(loc="lower right")
plt.axis([-5, 5, -200, 55])
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

# Step 02: Train the models with the training set with the same degrees:
weights_degree1 = pol_regression(x_train, y_train, degree=1)
weights_degree2 = pol_regression(x_train, y_train, degree=2)
weights_degree3 = pol_regression(x_train, y_train, degree=3)
weights_degree6 = pol_regression(x_train, y_train, degree=6)
weights_degree10 = pol_regression(x_train, y_train, degree=10)

# Compute the RMSE on the training set:
RMSE_degree1 = eval_pol_regression(parameters=weights_degree1, x=x_train, y=y_train, degree=1)
RMSE_degree2 = eval_pol_regression(parameters=weights_degree2, x=x_train, y=y_train, degree=2)
RMSE_degree3 = eval_pol_regression(parameters=weights_degree3, x=x_train, y=y_train, degree=3)
RMSE_degree6 = eval_pol_regression(parameters=weights_degree6, x=x_train, y=y_train, degree=6)
RMSE_degree10 = eval_pol_regression(parameters=weights_degree10, x=x_train, y=y_train, degree=10)

errors_train = [RMSE_degree1, RMSE_degree2, RMSE_degree3, RMSE_degree6, RMSE_degree10]
degrees = [1, 2, 3, 6, 10]

plt.plot(degrees, errors_train, color='red', label='Train Error')

# Compute the RMSE on the test set:
RMSE_degree1 = eval_pol_regression(parameters=weights_degree1, x=x_test, y=y_test, degree=1)
RMSE_degree2 = eval_pol_regression(parameters=weights_degree2, x=x_test, y=y_test, degree=2)
RMSE_degree3 = eval_pol_regression(parameters=weights_degree3, x=x_test, y=y_test, degree=3)
RMSE_degree6 = eval_pol_regression(parameters=weights_degree6, x=x_test, y=y_test, degree=6)
RMSE_degree10 = eval_pol_regression(parameters=weights_degree10, x=x_test, y=y_test, degree=10)

errors_test = [RMSE_degree1, RMSE_degree2, RMSE_degree3, RMSE_degree6, RMSE_degree10]
degrees = [1, 2, 3, 6, 10]  

# Plot the graph:
plt.plot(degrees, errors_test, color='blue', label='Test Error')
plt.legend(loc="lower left")
plt.show()

# %%


