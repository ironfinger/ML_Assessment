#%%
import numpy as np
import pandas as pd
import os

path = os.path.join('dog_breeds.csv')

data = pd.read_csv(path)
data.head()
easy_D = pd.read_csv('kmeans.csv', header = None)

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


def kmeans(dataset, k):
    
    # Initialisation:
    data_m = dataset.values # Store the entire dataset in a matrix.
    cluster_assigned = np.zeros((len(data_m), 2)) # Stores the assigned cluster and the distance. 
    
    # Step 01: Creation of variable K is done in method call.
    
    # Step 02: Randomly select 3 distinct centroids:
    centroids = initialise_centroids(dataset=dataset, k=k)
    
    # Step 03: Measure the distance between each point and the centroids:
    for v_index, vector in enumerate(data_m): # Loop through each row in the dataset
        
        centroid_distances = np.zeros(len(centroids)) # Array to store the distances.
        
        # Loop through the centroids to calculate the distances:
        for centroid_i, centroid in enumerate(centroids):
            distance = compute_euclidean_distance(vec_1=vector, vec_2=centroid)
            centroid_distances[centroid_i] = distance 
            
        # Step 04: Assign each point to the nearest cluster:
        min_distance = np.min(centroid_distances)
        assigned_cluster = np.argmin(centroid_distances)
        
        cluster_assigned[v_index] = np.array([min_distance, assigned_cluster])
        
    # Step 05: Calculate the mean of each cluster as the new centroids:
    
    # Create a new df with the assigned cluster as a new column:
    cluster_assigned_df = pd.DataFrame(data=dataset) 
    
    print(cluster_assigned)
    
    print(centroids)
    print(cluster_assigned_df)

#%%
kmeans(dataset=data, k=2)
# %%
