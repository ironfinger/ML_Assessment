#%%
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv('dog_breeds.csv')
easy_D = pd.read_csv('kmeans.csv', header = None)


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
    return centroids # Return the centroids.

def kmeans(dataset, k):

    # Initialise centroids:
    centroids = initialise_centroids(dataset=dataset, k=k)
    print('The centroids are: ', centroids)
    
    # Measure the distance:
    # Measure the distance between two random datapoints:
    distance = compute_euclidean_distance(centroids[0], centroids[1])
    print('The distance is: ', distance) 
    cluster_assigned = 0
    return 'hello'

h = kmeans(data, 2)
# %%
