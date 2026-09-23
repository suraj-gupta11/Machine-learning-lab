import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_and(X, Y, lr=0.05, epochs=1000):

    # 2 inputs -> 2 hidden -> 1 output
    w1 = np.random.randn(2, 2) * 0.1
    w2 = np.random.randn(2) * 0.1

    errors = []

    for epoch in range(epochs):

        total_error = 0

        for x, target in zip(X, Y):

            # Forward
            hidden = sigmoid(x @ w1)
            output = sigmoid(hidden @ w2)

            # Output error
            e = target - output

            # Output delta
            delta_output = e * output * (1 - output)

            # Hidden delta
            delta_hidden = (
                hidden * (1 - hidden) *
                w2 * delta_output
            )

            # Update weights
            w2 += lr * delta_output * hidden
            w1 += lr * np.outer(x, delta_hidden)

            total_error += e ** 2

        errors.append(total_error / 2)

        if errors[-1] <= 0.002:
            break

    return w1, w2, epoch + 1, errors


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([0, 0, 0, 1])

w1, w2, epochs, errors = train_and(X, Y)

print("Hidden weights:")
print(w1)

print("Output weights:")
print(w2)

print("Epochs:", epochs)