#%%
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv('dog_breeds.csv')
easy_D = pd.read_csv('kmeans.csv', header = None)
data.head()

# %%

def compute_euclidean_distance(vec_1, vec_2):

    total = 0 # Store the total distance

    # Loop through the vectors dimensions:
    for i, value in enumerate(vec_1):
        # Calculate the brackets and square them:
        total += (vec_2[i] - value) ** 2

    # Calculate the square root to get the final distance:
    distance = np.sqrt(total)
    return distance

def initialise_centroids(dataset, k):
    # Get a numpy representation of the dataset:
    df_numpy = dataset.values
    
    # Randomise the dataset:
    dataset_temp = df_numpy # Create a temporay dataset (just for my ocd).
    np.random.shuffle(dataset_temp) # Shuffle the dataset.
    centroids = dataset_temp[:k, :] # Retrieve the k nubmer of rows.
    #print('initialise_centroids', centroids)
    return centroids # Return the centroids.

def kmeans(dataset, k):

    # Initialise centroids:
    centroids = initialise_centroids(dataset=dataset, k=k) # WORKING
    
    # So now that we have initialised the centroids, we need to go through the dataset and measer the distance:
    data_matrix = dataset.values
    
    # We need a way to store the ground values:
    cluster_assigned = np.zeros((len(data_matrix), 2))

    for v_index, vector in enumerate(data_matrix):
        centroid_distances = np.zeros(len(centroids))

        # Loop through the centroids and calculate the distance between the centroid and the current vector:
        for c_index, centroid in enumerate(centroids):
            distance = compute_euclidean_distance(vec_1=vector, vec_2=centroid)
            centroid_distances[c_index] = distance

        # Now we need to store the smallest values in centroid distances:
        min_value = np.min(centroid_distances)
        min_index = np.argmin(centroid_distances)

        cluster_assigned[v_index] = np.array([min_value, min_index])
    

    # need to figure out the mean fml:
    temp_df = pd.DataFrame(data=dataset)
    temp_df['label_kmeans'] = cluster_assigned[:, 1]

    new_centroids = np.zeros([k, len(centroids[0])])

    for i, centroid in enumerate(centroids):

        # Get rid of the label:
        centroid_withoutk = centroid[:-1]
        group = temp_df[temp_df['label_kmeans'] == i]
        new_centroids[i, (len(centroids[0]) - 1)] = i

        # We need to get the group as a matrix:
        group_np = group.to_numpy()

        for x, val in enumerate(centroid_withoutk):
            column = group_np[:, x]
            mean = np.mean(column)
            new_centroids[i, x] = mean

    return centroids, new_centroids, cluster_assigned



centroids, mean_c, cluster_assigned = kmeans(dataset=data, k=2)





#%%
centroids, meann_c,  cluster_assigned = kmeans(data, 2)

# print('The Centroids: ', centroids)
# print('Cluster Assigned', cluster_assigned)
cluster_groups = cluster_assigned[:, 1]
distance = cluster_assigned[:, 0]

# Assign the clusters in the data frame:
new_df = pd.DataFrame(data=data)

new_df['label_kmeans'] = cluster_groups
new_df['distance'] = distance
new_df.head()

#%%
centroid_update = new_df[new_df['distance'] == 0]
centroid_update.head()

#%%
# Display the data and graphs:
import matplotlib.pyplot as plt
plt.scatter(
    new_df['height'][new_df['label_kmeans'] == 0],
    new_df['tail length'][new_df['label_kmeans'] == 0],
    color='blue'
)
plt.scatter(
    new_df['height'][new_df['label_kmeans'] == 1],
    new_df['tail length'][new_df['label_kmeans'] == 1],
    color='red'
)
plt.scatter(
    new_df['height'][new_df['distance'] == 0],
    new_df['tail length'][new_df['distance'] == 0],
    color='black',
    s=100,
)
plt.scatter(
    mean_c[0][0],
    meann_c[0][1],
    c='yellow',
    s=200
)
plt.scatter(
    mean_c[1][0],
    meann_c[1][1],
    c='yellow',
    s=200
)
plt.show()
# %%
