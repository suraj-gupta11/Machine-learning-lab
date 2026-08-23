import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_predict(X_train, y_train, x_test, k=3):
    distances = [
        euclidean_distance(x, x_test)
        for x in X_train
    ]

    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]

    return Counter(nearest_labels).most_common(1)[0][0]