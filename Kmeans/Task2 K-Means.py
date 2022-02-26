#%%
from xml.parsers.expat import errors
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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
    dataset_temp = df_numpy # Create a temporay dataset.
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
        
"""Does Convergence !!!!"""
def kmeans(dataset, k):
    # Step 01: Creation of variable K is done in method call.
    
    # Initialisation:
    data_m = dataset.values # Store the entire dataset in a matrix.
    cluster_assigned = np.zeros((len(data_m), 2)) # Stores the assigned cluster and the distance. 
    errors_per_cycle = []
    cycles = []
    current_cycle = 0
    
    # Step 02: Randomly select 3 distinct centroids:
    centroids = initialise_centroids(dataset=dataset, k=k)
    
    # Step 03 Measure the distance between each point and the centroids + Step 04: Assign each point to the nearest cluster:
    cluster_assigned = assign_clusters(data_m=data_m, centroids=centroids)
    
    # Compute the objective function error:
    objective_function_error = np.sum(cluster_assigned[:, 0])
    current_cycle += 1
    
    errors_per_cycle.append(objective_function_error)    
    cycles.append(current_cycle)
        
    # Create a new df with the assigned cluster as a new column:
    cluster_assigned_df = dataset.copy()
    cluster_assigned_df['assigned_centroid'] = cluster_assigned[:, 1]
    
    # Step 05: Calculate the mean of each cluster as the new centroids:
    new_centroids = compute_mean_centroids(cluster_assigned_df=cluster_assigned_df, centroids=centroids, k=k)
    
    # Assign the mean centroids to become the new centroids:
    centroids = new_centroids
    
    convergence = False
    
    while convergence == False:
        # Compute step 03 and 04 again:
        cluster_assigned = assign_clusters(data_m=data_m, centroids=centroids)
        
        objective_function_error = np.sum(cluster_assigned[:, 0])
        current_cycle += 1
        errors_per_cycle.append(objective_function_error)
        cycles.append(current_cycle)
        
        # Create a new df with the assigned cluster as a new column:
        cluster_assigned_df = dataset.copy()
        cluster_assigned_df['assigned_centroid'] = cluster_assigned[:, 1]
        
        # Step 05: Calculate the mean of each cluster as the new centroids:
        new_centroids = compute_mean_centroids(cluster_assigned_df=cluster_assigned_df, centroids=centroids, k=k)
    
        # Now we need to check if the centroids are equal:
        if np.array_equal(centroids, new_centroids):
            convergence = True
            
            plt.plot(cycles, errors_per_cycle)
            plt.title('KMeans Objective Function Error')
            plt.xlabel('Cycle')
            plt.ylabel('Error')
            plt.show()
            centroids = new_centroids
            
            return centroids, cluster_assigned_df 
        else:
            centroids = new_centroids
            
            
    return centroids, cluster_assigned_df

#%%
centroids, cluster_assigned_df = kmeans(dataset=data, k=2)
# %%

# Time to plot the graphs:
k_means_1 = cluster_assigned_df[cluster_assigned_df['assigned_centroid'] == 0]
k_means_2 = cluster_assigned_df[cluster_assigned_df['assigned_centroid'] == 1]

import matplotlib.pyplot as plt

# PLot the graph:

# %%

plt.title('K-Means Cluster graph between Height and Tail length')

# K = 2 For Height and Tail Length:
plt.scatter(
    k_means_1['height'],
    k_means_1['tail length'],
    label='K = 1',
    color='blue'
)

# Plot the second group:
plt.scatter(
    k_means_2['height'],
    k_means_2['tail length'],
    label='K = 2',
    color='red'
)

# Plot the first centroids height = [0] tail length = [1]:
plt.scatter(
    centroids[0, 0],
    centroids[0, 1],
    label='K = 1 Centroid',
    color='yellow'
)

# Plot the second centroid:
plt.scatter(
    centroids[1, 0],
    centroids[1, 1],
    label='K = 2 Centroid',
    color='green'
)
plt.legend(loc='upper right')
plt.show()


# K = 2 For Height and Leg Length
plt.title('K-Means Cluster graph between Height and Leg Length')
plt.scatter(
    k_means_1['height'],
    k_means_1['leg length'],
    label='K = 1',
    color='blue'
)

# Plot the second group:
plt.scatter(
    k_means_2['height'],
    k_means_2['leg length'],
    label='K = 2',
    color='red'
)

# Plot the first centroids height = [0] tail length = [1]:
plt.scatter(
    centroids[0, 0],
    centroids[0, 2],
    label='K = 1 Centroid',
    color='yellow'
)

# Plot the second centroid:
plt.scatter(
    centroids[1, 0],
    centroids[1, 2],
    label='K = 2 Centroid',
    color='green'
)
plt.legend(loc='lower right')
plt.show()

#%%
# Commence K means for K = 3
centroids, cluster_assigned_df = kmeans(dataset=data, k=3)

# Time to plot the graphs:
k_means_1 = cluster_assigned_df[cluster_assigned_df['assigned_centroid'] == 0]
k_means_2 = cluster_assigned_df[cluster_assigned_df['assigned_centroid'] == 1]
k_means_3 = cluster_assigned_df[cluster_assigned_df['assigned_centroid'] == 2]

# K = 3 For Height and Tail Length:
plt.title('K-Means Cluster graph bettwen Height and Tail Length')
plt.scatter(
    k_means_1['height'],
    k_means_1['tail length'],
    label='K = 1',
    color='blue'
)

# Plot the second group:
plt.scatter(
    k_means_2['height'],
    k_means_2['tail length'],
    label='K = 2',
    color='red'
)

# Plot the third group:
plt.scatter(
    k_means_3['height'],
    k_means_3['tail length'],
    label='K = 3',
    color='green'
)

# Plot the first centroids height = [0] tail length = [1]:
plt.scatter(
    centroids[0, 0],
    centroids[0, 1],
    label='K = 1 Centroid',
    color='yellow'
)

# Plot the second centroid:
plt.scatter(
    centroids[1, 0],
    centroids[1, 1],
    label='K = 2 Centroid',
    color='purple'
)

plt.scatter(
    centroids[2, 0],
    centroids[2, 1],
    label='K = 3 Centroid',
    color='orange'
)
plt.legend(loc='upper right')
plt.show()

# K = 3 For Height and Leg Length:
plt.title('K-Means Cluster graph between Height and Leg Length')
plt.scatter(
    k_means_1['height'],
    k_means_1['leg length'],
    label='K = 1',
    color='blue'
)

# Plot the second group:
plt.scatter(
    k_means_2['height'],
    k_means_2['leg length'],
    label='K = 2',
    color='red'
)

# Plot the third group:
plt.scatter(
    k_means_3['height'],
    k_means_3['leg length'],
    label='K = 3',
    color='green'
)

# Plot the first centroids height = [0] tail length = [1]:
plt.scatter(
    centroids[0, 0],
    centroids[0, 2],
    label='K = 1 Centroid',
    color='yellow'
)

# Plot the second centroid:
plt.scatter(
    centroids[1, 0],
    centroids[1, 2],
    label='K = 2 Centroid',
    color='purple'
)

plt.scatter(
    centroids[2, 0],
    centroids[2, 2],
    label='K = 3 Centroid',
    color='orange'
)
plt.legend(loc='lower right')
plt.show()
