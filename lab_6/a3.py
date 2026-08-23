import time
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.neighbors import KNeighborsClassifier


def evaluate_model(model, X_train, X_test, y_train, y_test):

    # Accuracy, Precision, Recall and F1
    y_pred = model(X_train, y_train, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Average execution time over 10 runs
    times = []

    for _ in range(10):
        start = time.perf_counter()

        model(X_train, y_train, X_test)

        end = time.perf_counter()

        times.append(end - start)

    avg_time = np.mean(times)

    return accuracy, precision, recall, f1, avg_time