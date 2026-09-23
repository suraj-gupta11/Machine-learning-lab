import numpy as np


def bipolar_step(y):
    return 1 if y >= 0 else -1


def sigmoid(y):
    return 1 / (1 + np.exp(-y))


def relu(y):
    return max(0, y)


def train(X, Y, weights, lr, activation, max_epochs=1000):
    for epoch in range(max_epochs):

        total_error = 0

        for x, target in zip(X, Y):

            y = activation(np.dot(x, weights))

            # Convert bipolar output for comparison
            if activation == bipolar_step:
                y = 1 if y == 1 else 0

            e = target - y
            weights = weights + lr * e * x

            total_error += e ** 2

        if total_error / 2 <= 0.002:
            return weights, epoch + 1

    return weights, max_epochs


X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

Y = np.array([0, 0, 0, 1])

for name, activation in [
    ("Bipolar Step", bipolar_step),
    ("Sigmoid", sigmoid),
    ("ReLU", relu)
]:

    w = np.array([10.0, 0.2, -0.75])

    final_w, epochs = train(
        X, Y, w, 0.05, activation
    )

    print(name, ":", epochs, "epochs")