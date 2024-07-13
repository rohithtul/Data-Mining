import sys
import numpy as np
import matplotlib.pyplot as plt

#input file name read from command line
training_file = sys.argv[1]
k_values = range(2, 11)

# Read data from file and dropping last column
dataset = np.loadtxt(training_file)
# Remove the last column from the array
dataset = dataset[:, :-1]


def kmeans(dataset, k, max_iterations=100):
    # Initialize centroids and assign clusters
    centroids = init_center(dataset, k)
    for iteration in range(max_iterations):
        class_labels = allocate_clusters(dataset, centroids)
        # Update centroids based on the new cluster assignments
        new_centroids = upd_center(dataset, class_labels, k)
        # If the centroids have not changed, stop the algorithm
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    # Compute the final SSE
    sse = 0
    for i in range(k):
        cluster_points = dataset[class_labels == i]
        centroid = centroids[i]
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse


def allocate_clusters(dataset, centroids):
    # Assign each data point to the nearest centroid
    n_points = dataset.shape[0]
    n_clusters = centroids.shape[0]
    class_labels = np.zeros(n_points)
    for i in range(n_points):
        distances = np.zeros(n_clusters)
        for j in range(n_clusters):
            distances[j] = L2_distance(dataset[i], centroids[j])
        class_labels[i] = np.argmin(distances)
    return class_labels

def init_center(dataset, k):
    # random-partition intialization
    random_points = np.random.choice(dataset.shape[0], k, replace=False)
    centroids = dataset[random_points]
    return centroids

def upd_center(dataset, class_labels, k):
    # Compute the new centroids as the mean of the data points in each cluster
    n_features = dataset.shape[1]
    centroids = np.zeros((k, n_features))
    for i in range(k):
        cluster_points = dataset[class_labels == i]
        centroids[i] = np.mean(cluster_points, axis=0)
    return centroids

def L2_distance(x, y):
    sum_square = np.sum((x - y) ** 2)
    l2_distance = np.sqrt(sum_square)
    return l2_distance

# Compute SSE for different k values
sse_errors = []
for k in k_values:
    sse = kmeans(dataset, k, 20)
    sse_errors.append(sse)
    print("For k = %d After 20 iterations: SSE error = %.4f" % (k, sse))

# Plot SSE vs k using the elbow method
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(k_values, sse_errors,'bx-')
ax.set_xlabel(' k_values ')
ax.set_ylabel('sse_errors ')
ax.set_title('SSE vs K chart for '+training_file)
plt.show()
