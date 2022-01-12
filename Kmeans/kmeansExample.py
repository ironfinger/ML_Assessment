"""
K-means algorithm
"""

#%%

# Import libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %%

# Generate some random data:
X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
plt.show()

# %%

Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

# %%

# Store the coords of the cluster centers:
cluster_points = Kmean.cluster_centers_

# %%

# Display the clusters: 
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
plt.scatter(1.82471261, 2.1636134, s=200, c='g', marker='s')
plt.scatter(-0.99642714, -1.10158733, s=200, c='r', marker='s')

# %%

# Testing the algorithm:
labels = Kmean.labels_
print(labels)
# %%

# Predict the cluster of a data point:
sample_test = np.array([-3.0, -3.0])
seconda_test = sample_test.reshape(1, -1)
print(Kmean.predict(seconda_test))

# %%
