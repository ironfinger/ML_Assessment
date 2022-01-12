# K Means is one of the simplest and popular unsupervised machine learning things.

- The objective of K-means is simple: group similar data points together and discover underlying patters. To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset. 
    - A cluster referes to a collection of data points aggregated together because of certain similarities.

- You will define a target number k, which referes  to the number of centroids you need in the datset. A ventroid is the imahinary or real location representing the center of the cluster.
    - Each data point is allocated to each of the clusters through reducing the incluster sum of squares.

- In other words, the K-means algorithm identifies k number of centroids and then allocates every data point to the nearest cluster while keeping the centroids as small as possible.

## The 'means' in K-means refers to averaging of the data; that is finding the centroid.

# How it works:
- The algorithm starts with a first group of randomly selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) calculations to optimize the positions of the centroids.

- This creating/optimisation of clusters stops when:
    - The centroids have stabilized - there is no change in their values because the clustering has been successful.
    - The defined number of iterations has been achieved.


# Second Article: https://medium.com/data-folks-indonesia/step-by-step-to-understanding-k-means-clustering-and-implementation-with-sklearn-b55803f519d6

## K-means clustering is a simple and elegant approach for partitioning a data set into K distinct, non-overlapping clusters.
- In order to perform K-means clustering, we must first specify the desired number of cluesters K; then the K-means algorithm will assign each observation to exactly one of the K-clusters.

### K-means clusters data by trying to separate samples in n groups of equal variance.
    - This therefore minimises a criterion known as the inertia or whithin-cluster sum of sqaures.
    - The K-means aims to choose centroid that minimises the inertia or whithin-cluster sum of squares criterion.

## How does it work:
### Step1: Determine the value 'k' -> This is the number of clusters.
### Step2: Randomly select 3 distinc centroid (new data points as cluster initialization)
### Step3: Measure the distance (euclidean distance) between each point and the centroid. 
    - For example measure the distance between first point and the centroid.
### Step4: Assign each point to the nearest cluster.
### Step5: Calculate the mean of each cluster as new centroid.
### Step 6: Repeat step 3-5 with the new center of cluster.
    - Repeat until stop -> Convergence (no further changes) | Maximum number of interations.
### Step 7: Assess the results of this clustering via calculating the variance of each cluster.

    """
    The euclidean distance between vector a = [4, 5] b = [6, 7]:
    therefore: sqrt((4 - 6)^2 + (5 - 7)^2)

    therefore for 3d space: a = [4, 5, 6] b = [7, 8, 9]
    thereforeL sqrt((4 - 7)^2 + (5 - 8)^2 + (6 - 9)^2)
    """
