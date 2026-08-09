
import numpy as np
import pandas as pd


# Label Encoding
def label_encode(col):

    values = sorted(col.dropna().unique())

    result = []

    for x in col:
        if pd.isna(x):
            result.append(-1)
        else:
            for i in range(len(values)):
                if x == values[i]:
                    result.append(i)

    return np.array(result), values


# One Hot Encoding
def one_hot_encode(col):

    values = sorted(col.dropna().unique())

    result = np.zeros((len(col), len(values)))

    for i in range(len(col)):

        if pd.isna(col.iloc[i]):
            continue

        for j in range(len(values)):

            if col.iloc[i] == values[j]:
                result[i][j] = 1

    return result, values


# Minkowski Distance
def minkowski_dist(a, b, p):

    total = 0

    for i in range(len(a)):
        total = total + abs(a[i] - b[i]) ** p

    return total ** (1 / p)


# Dot Product
def my_dot(a, b):

    total = 0

    for i in range(len(a)):
        total = total + a[i] * b[i]

    return total


# Norm
def my_norm(v):

    total = 0

    for x in v:
        total = total + x * x

    return total ** 0.5


# Mean
def my_mean(data):

    total = 0

    for x in data:
        total = total + x

    return total / len(data)


# Variance
def my_var(data):

    mean = my_mean(data)

    total = 0

    for x in data:
        total = total + (x - mean) ** 2

    return total / len(data)


# Standard Deviation
def my_std(data):

    return my_var(data) ** 0.5


# K-Means
def kmeans(data, k, max_iters=100, seed=1):

    data = np.array(data, dtype=float)

    np.random.seed(seed)

    n = len(data)

    # Select random centroids
    index = np.random.choice(n, k, replace=False)

    centroids = data[index].copy()

    labels = np.zeros(n, dtype=int)

    for iteration in range(max_iters):

        new_labels = []

        # Assign points to clusters
        for i in range(n):

            distances = []

            for j in range(k):

                distance = 0

                for f in range(data.shape[1]):

                    distance = distance + (
                        data[i][f] - centroids[j][f]
                    ) ** 2

                distance = distance ** 0.5

                distances.append(distance)

            small = distances[0]
            cluster = 0

            for j in range(1, k):

                if distances[j] < small:
                    small = distances[j]
                    cluster = j

            new_labels.append(cluster)

        new_labels = np.array(new_labels)

        # Calculate new centroids
        new_centroids = []

        for j in range(k):

            points = data[new_labels == j]

            if len(points) > 0:
                center = np.mean(points, axis=0)
            else:
                center = centroids[j]

            new_centroids.append(center)

        new_centroids = np.array(new_centroids)

        # Check convergence
        if np.array_equal(labels, new_labels):

            labels = new_labels
            centroids = new_centroids

            break

        labels = new_labels
        centroids = new_centroids

    return labels, centroids


# --------------------------------------------------
# TESTING THE FUNCTIONS
# --------------------------------------------------

if __name__ == "__main__":

    print("Testing Functions")
    print("-----------------")

    # Data for testing
    a = [1, 2, 3]
    b = [4, 5, 6]

    print("\nMinkowski Distance:")
    print(minkowski_dist(a, b, 2))

    print("\nDot Product:")
    print(my_dot(a, b))

    print("\nNorm:")
    print(my_norm(a))

    numbers = [10, 20, 30, 40, 50]

    print("\nMean:")
    print(my_mean(numbers))

    print("\nVariance:")
    print(my_var(numbers))

    print("\nStandard Deviation:")
    print(my_std(numbers))

    # Label encoding test
    data = pd.Series(["Red", "Blue", "Green", "Red", "Blue"])

    encoded, values = label_encode(data)

    print("\nLabel Encoding:")
    print(encoded)

    print("Values:")
    print(values)

    # One hot encoding test
    encoded2, values2 = one_hot_encode(data)

    print("\nOne Hot Encoding:")
    print(encoded2)

    print("Values:")
    print(values2)

    # K-Means test
    data2 = [
        [1, 2],
        [1, 3],
        [2, 2],
        [8, 8],
        [9, 8],
        [8, 9]
    ]

    labels, centroids = kmeans(data2, 2)

    print("\nK-Means Labels:")
    print(labels)

    print("\nK-Means Centroids:")
    print(centroids)
