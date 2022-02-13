#%%
import numpy as np
import pandas as pd
import os

from sklearn import cluster

path = os.path.join('dog_breeds.csv')

data = pd.read_csv(path)
easy_D = pd.read_csv('kmeans.csv', header = None)

#%%
data.head()
#%%

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


# This function assigns each row to a new cluster:
def assign_clusters(data_m, centroids):
    cluster_assigned = np.zeros((len(data_m), 2))
    
    # Step 03: Measure the distance between each point and the centroids:
    for v_index, vector in enumerate(data_m):
        
        centroid_distances = np.zeros(len(centroids)) # Array to store the distances.
        
        # Loop through the centoirds to calculate the distances:
        for centroid_i, centroid in enumerate(centroids):
            distance = compute_euclidean_distance(vec_1=vector, vec_2=centroid)
            centroid_distances[centroid_i] = distance
            
        # Step 04: Assign each point to the nearest cluster:
        min_distance = np.min(centroid_distances)
        assigned_cluster = np.argmin(centroid_distances)
        
        cluster_assigned[v_index] = np.array([min_distance, assigned_cluster])
    
    return cluster_assigned


# This function creates the mean centroids to re-calculate the new clusters with:
def compute_mean_centroids(cluster_assigned_df, centroids, k):
    new_centroids = np.zeros([k, len(centroids[0])]) # Create an empty array to store the new centroids.
    
    for centroid_i, centroid in enumerate(centroids):
        
        # Copy the cluster assigned df so we don't make changes to both dataframes:
        current_centroid = cluster_assigned_df.copy()
        
        # Select all the rows where the assigned centroid is equal to the current centroid being assessed:
        current_centroid = current_centroid[current_centroid['assigned_centroid'] == centroid_i]
        
        # Next we need to drop the assigned_centroid column to calculate the mean centroid in that group:
        current_centroid = current_centroid.drop(['assigned_centroid'], axis=1)
        
        # Next I put the current centroid group into a numpy matrix:
        current_group_np = current_centroid.values
        
        # Next we need to calcualte the mean centroid:
        for x, val in enumerate(centroid):
            # Get the current column in question:
            current_column = current_group_np[:, x]
            
            # Calculate the mean of that column:
            mean = np.mean(current_column)
            
            new_centroids[centroid_i, x] = mean
            
    return new_centroids
        
        
    
def kmeans(dataset, k):
    # Step 01: Creation of variable K is done in method call.
    
    # Initialisation:
    data_m = dataset.values # Store the entire dataset in a matrix.
    cluster_assigned = np.zeros((len(data_m), 2)) # Stores the assigned cluster and the distance. 
    
    # Step 02: Randomly select 3 distinct centroids:
    centroids = initialise_centroids(dataset=dataset, k=k)
    
    # Step 03 Measure the distance between each point and the centroids + Step 04: Assign each point to the nearest cluster:
    cluster_assigned = assign_clusters(data_m=data_m, centroids=centroids)
        
    # Create a new df with the assigned cluster as a new column:
    cluster_assigned_df = dataset.copy()
    cluster_assigned_df['assigned_centroid'] = cluster_assigned[:, 1]
    
    # Step 05: Calculate the mean of each cluster as the new centroids:
    new_centroids = compute_mean_centroids(cluster_assigned_df=cluster_assigned_df, centroids=centroids, k=k)
        
    return centroids, new_centroids, cluster_assigned_df

def kmeansV2(dataset, k):
    # Step 01: Creation of variable K is done in method call.
    
    # Initialisation:
    data_m = dataset.values # Store the entire dataset in a matrix.
    cluster_assigned = np.zeros((len(data_m), 2)) # Stores the assigned cluster and the distance. 
    
    # Step 02: Randomly select 3 distinct centroids:
    centroids = initialise_centroids(dataset=dataset, k=k)
    
    # Step 03 Measure the distance between each point and the centroids + Step 04: Assign each point to the nearest cluster:
    cluster_assigned = assign_clusters(data_m=data_m, centroids=centroids)
        
    # Create a new df with the assigned cluster as a new column:
    cluster_assigned_df = dataset.copy()
    cluster_assigned_df['assigned_centroid'] = cluster_assigned[:, 1]
    
    # Step 05: Calculate the mean of each cluster as the new centroids:
    new_centroids = compute_mean_centroids(cluster_assigned_df=cluster_assigned_df, centroids=centroids, k=k)
    
    # Assign the mean centroids to become the new centroids:
    centroids = new_centroids
    
    epochs = 10
    cycle = 0
    
    print('Starting while loop')
    for i in range(0, epochs):
        
        
        
        print('Current cylce: ', i)
        
        # Compute step 03 and 04 again:
        cluster_assigned = assign_clusters(data_m=data_m, centroids=centroids)
        
        # Create a new df with the assigned cluster as a new column:
        cluster_assigned_df = dataset.copy()
        cluster_assigned_df['assigned_centroid'] = cluster_assigned[:, 1]
        
        # Step 05: Calculate the mean of each cluster as the new centroids:
        new_centroids = compute_mean_centroids(cluster_assigned_df=cluster_assigned_df, centroids=centroids, k=k)
        
        # # Now we need to check for changes:
        # if new_centroids == centroids:
        #     centroids = new_centroids
        #     cylce = 10
            
        # else:
        #     cycle += 1
        #     centroids = new_centroids
        
        # Make the new centroids the current centroids:
        centroids = new_centroids

    return centroids, new_centroids, cluster_assigned_df
        
    


#%%
centroids, mean_centroids, cluster_assigned_df = kmeansV2(dataset=data, k=2)



# %%

# Time to plot the graphs:
k_means_1 = cluster_assigned_df[cluster_assigned_df['assigned_centroid'] == 0]
k_means_2 = cluster_assigned_df[cluster_assigned_df['assigned_centroid'] == 1]

import matplotlib.pyplot as plt

# PLot the graph:

# Plot the first group:
plt.scatter(
    k_means_1['height'],
    k_means_1['tail length'],
    color='blue'
)

# Plot the second group:
plt.scatter(
    k_means_2['height'],
    k_means_2['tail length'],
    color='red'
)

# Plot the first centroids height = [0] tail length = [1]:
plt.scatter(
    centroids[0, 0],
    centroids[0, 1],
    color='black'
)

# Plot the second centroid:
plt.scatter(
    centroids[1, 0],
    centroids[1, 1],
    color='black'
)

# Plot the first mean centroid:
plt.scatter(
    mean_centroids[0, 0],
    mean_centroids[1, 1],
    color='yellow'
)

# Plot the mean centroids:
plt.scatter(
    mean_centroids[1, 0],
    mean_centroids[1, 1,],
    color='yellow'
)

plt.show()

# %%

# The next step on the TODO list is to repeat the mean cluster setup with new centroids:
# We can leave the kmean algorithm as it is but re-write the mean stuff into a new function which we can,
# loop until convergence, or max number of iterations will be 10 for now.
# If that works, then we can look at thinking abou the variance -> Not sure about this.
# Then regress and evaluate.
