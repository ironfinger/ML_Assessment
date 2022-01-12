"""
Cluster the dog breed data
"""

#%%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


#%%

# Import the data:
data_frame = pd.read_csv('dog_breeds.csv')
data_frame.head()

# %%

# Get columns as arrays:
height = data_frame['height']
tail_length = data_frame['tail length']
leg_length = data_frame['leg length']
nose_circum = data_frame['nose circumference']

plt.scatter(height, tail_length, color='red')
plt.scatter(height, leg_length, color='pink')
plt.scatter(height, nose_circum, color='purple')
plt.show()

# %%

# The data includes 4 features:
#height, tail length, leg_length, nose_circumference
kmeans = KMeans(n_clusters=3).fit(data_frame)
labels = kmeans.labels_
print(labels)
# %%

# Create a new table with the labels:
new_data_frame = pd.DataFrame(data=data_frame)
new_data_frame['label_kmeans'] = labels
new_data_frame.head()

# %%

# Plot k-mean:
# height -> tail length
plt.scatter(
    new_data_frame['height'][new_data_frame["label_kmeans"] == 0],
    new_data_frame['tail length'][new_data_frame["label_kmeans"] == 0],
    color='blue'    
)
plt.scatter(
    new_data_frame['height'][new_data_frame["label_kmeans"] == 1],
    new_data_frame['tail length'][new_data_frame["label_kmeans"] == 1],
    color='green'    
)
plt.scatter(
    new_data_frame['height'][new_data_frame["label_kmeans"] == 2],
    new_data_frame['tail length'][new_data_frame["label_kmeans"] == 2],
    color='red'    
)
plt.show()
# height -> leg length

# 
# %%
