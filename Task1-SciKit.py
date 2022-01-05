#%% 
import os
from numpy.core.fromnumeric import size
import pandas as pd

dataset_path = os.path.join('data', 'polyRegression.csv')
df = pd.read_csv(dataset_path)

df.head()
# %%

# Create X and Y variables:
X = df['x']
Y = df['y']

print(X, Y)

#%%
# Split the training data
from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# %%

# View the data:
import matplotlib.pyplot as plt

plt.scatter(X_train, Y_train)
plt.show()

# %%

# View the test data:

plt.scatter(X_test, Y_test)
plt.show()
# %%

# Attempt the polynomial regression:

import numpy as np

"""
Polynomial regression related vocabulary:

tutorial -> https://data36.com/polynomial-regression-python-scikit-learn/

- Degree of a polynomial -> The highest power (largest exponent) in your polynomial
    - In the example case its 4 cus of 4 curves -> I think its 2 degree on this case.

- Coefficient: each number in our polynomial is a coefficient; these are the parameters that are unknown.
    - Out polynomial regression model will try to estimate these values.

- Leading term: the term with the highest power (in the example if 3x^4, for this it could be ?x^2).
    - This is the most important part of the polynomial as it determines the graphs behaviour.

- Leading coeficient: The number before x where the 'x' is the highest power.

- Constant term: This is the y intercept, it never changes... ever.

The official definition:

A mathematical expression is a polynomial if:
    - The expression has a finite number of terms.
    - Each term has a coefficient.
    - The coefficient is multiplied by a variable.
    - The variables are raised to a non-negatibe integer power.

Polynomial vs linear regression:

Polynomial standard form -> 3x^4 - 7x^3 + 2x^2 + 11
Polynomial MachineL form -> y = B[0] + B[1]x + B[2]x^2 + ... + B[n]x^n

y -> Is the response we want to predict.
x -> Is the feature.
B[0] -> Is the intercept.
B[n]x^n -> Are the coefficient we'd like to find when we train the model.
n is the degree of the polynomial ( the high n is, the more complex curved lines you can create).

"""

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)

x = np.array(X_train)

# This created the x values and x^2 values if you refer to the equation this is what we want.
poly_features = poly.fit_transform(x.reshape(-1, 1))
print(poly_features)

# Creating the regression model:
from sklearn.linear_model import LinearRegression
poly_reg_model = LinearRegression() # We use this because it is a linear model.

poly_reg_model.fit(poly_features, Y_train) # This is the model training.

y_predicted = poly_reg_model.predict(poly_features)
print(y_predicted)

# %%

# Data visualisation:
plt.figure(figsize=(10, 6))
plt.title("Poly", size=16)
plt.scatter(x, Y_train)
plt.plot(x, y_predicted, c="red")
plt.show()
# %%
