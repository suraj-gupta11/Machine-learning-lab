''' using AI python'''

import numpy as np
import pandas as pd


def label_encode(col):
    values = sorted(col.dropna().unique())

    mapping = {}

    for i in range(len(values)):
        mapping[values[i]] = i

    result = []

    for x in col:
        if pd.isna(x):
            result.append(-1)
        else:
            result.append(mapping[x])

    return np.array(result), mapping


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


def minkowski_dist(a, b, p):
    total = 0

    for i in range(len(a)):
        total = total + abs(a[i] - b[i]) ** p

    return total ** (1 / p)


def my_dot(a, b):
    total = 0

    for i in range(len(a)):
        total = total + a[i] * b[i]

    return total


def my_norm(v):
    total = 0

    for x in v:
        total = total + x * x

    return total ** 0.5


def my_mean(data):
    total = 0

    for x in data:
        total = total + x

    return total / len(data)


def my_var(data):
    mean = my_mean(data)
    total = 0

    for x in data:
        total = total + (x - mean) ** 2

    return total / len(data)


def my_std(data):
    return my_var(data) ** 0.5


def kmeans(data, k, max_iters=100, seed=1):

    data = np.array(data, dtype=float)

    np.random.seed(seed)

    n = len(data)

    index = np.random.choice(n, k, replace=False)

    centroids = data[index].copy()

    labels = np.zeros(n, dtype=int)

    for iteration in range(max_iters):

        new_labels = []

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

        new_centroids = []

        for j in range(k):

            points = data[new_labels == j]

            if len(points) > 0:
                center = np.mean(points, axis=0)
            else:
                center = centroids[j]

            new_centroids.append(center)

        new_centroids = np.array(new_centroids)

        if np.array_equal(labels, new_labels):
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    return labels, centroids
print("Program is running")

a = [1, 2, 3]
b = [4, 5, 6]

print("Minkowski Distance:", minkowski_dist(a, b, 2))
print("Dot Product:", my_dot(a, b))
print("Norm:", my_norm(a))

data = [10, 20, 30, 40, 50]

print("Mean:", my_mean(data))
print("Variance:", my_var(data))
print("Standard Deviation:", my_std(data))

x = pd.Series(["b", "a", "c", "a"])

encoded, mapping = label_encode(x)

print("Label Encoding:", encoded)
print("Mapping:", mapping)

encoded2, categories = one_hot_encode(x)

print("One Hot Encoding:")
print(encoded2)

data2 = [
    [1, 2],
    [1, 3],
    [2, 2],
    [8, 8],
    [9, 8],
    [8, 9]
]

labels, centroids = kmeans(data2, 2)

print("K-Means Labels:", labels)
print("K-Means Centroids:")
print(centroids)
