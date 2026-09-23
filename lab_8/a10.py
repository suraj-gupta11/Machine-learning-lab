import numpy as np


def step(x):
    return (x >= 0).astype(int)


def train(X, Y, weights, lr=0.05, epochs=1000):

    for epoch in range(epochs):

        error = 0

        for x, target in zip(X, Y):

            output = step(x @ weights)

            e = target - output

            weights += lr * np.outer(x, e)

            error += np.sum(e ** 2)

        if error / 2 <= 0.002:
            return weights, epoch + 1

    return weights, epochs


X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# AND -> two outputs
Y = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1]
])

weights = np.array([
    [10, 10],
    [0.2, 0.2],
    [-0.75, -0.75]
], dtype=float)

weights, epochs = train(X, Y, weights)

print("Final weights:")
print(weights)

print("Epochs:", epochs)