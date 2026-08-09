import pandas as pd
import numpy as np

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Select Numerical Features --------------------

data = df.select_dtypes(include=['int64', 'float64'])
data = data.fillna(0)

X = data.values

# -------------------- Euclidean Distance --------------------

def euclidean_distance(point1, point2):

    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2

    return distance ** 0.5

# -------------------- Assign Clusters --------------------

def assign_clusters(X, centroids):

    clusters = []

    for point in X:

        distances = []

        for centroid in centroids:
            distances.append(euclidean_distance(point, centroid))

        cluster = distances.index(min(distances))
        clusters.append(cluster)

    return clusters

# -------------------- Update Centroids --------------------

def update_centroids(X, clusters, k):

    centroids = []

    for i in range(k):

        cluster_points = []

        for j in range(len(X)):
            if clusters[j] == i:
                cluster_points.append(X[j])

        if len(cluster_points) == 0:
            centroids.append(X[np.random.randint(len(X))])
        else:
            centroids.append(np.mean(cluster_points, axis=0))

    return np.array(centroids)

# -------------------- K-Means Function --------------------

def k_means(X, k, max_iterations=100):

    np.random.seed(42)

    random_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[random_indices]

    for iteration in range(max_iterations):

        clusters = assign_clusters(X, centroids)

        new_centroids = update_centroids(X, clusters, k)

        if np.allclose(centroids, new_centroids):
            print("Converged after", iteration + 1, "iterations")
            break

        centroids = new_centroids

    return clusters, centroids

# -------------------- Run K-Means --------------------

k = 3

clusters, centroids = k_means(X, k)

# -------------------- Results --------------------

print("\nFinal Centroids:\n")
print(centroids)

print("\nFirst 20 Cluster Assignments:\n")
print(clusters[:20])

print("\nNumber of points in each cluster:\n")

for i in range(k):
    print(f"Cluster {i} :", clusters.count(i))