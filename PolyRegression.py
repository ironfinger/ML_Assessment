"""
What is polynomial regression:

"""

#%%
import pandas as pd

data_train = pd.read_csv('regression_train.csv')
data_test = pd.read_csv('regression_test.csv')

data_train.head()

# %%

x_train = data_train['x']
y_train = data_train['y']

x_test = data_test['x']
y_test = data_test['y']

print(x_train)
# %%

import matplotlib.pyplot as plt

plt.scatter(x_train, y_train, color='red')
plt.show()

# %%


# Prepare the data matrix:
import numpy as np
Xtilde = np.column_stack((np.ones(x_train.shape), x_train))
print(Xtilde)


# %%

