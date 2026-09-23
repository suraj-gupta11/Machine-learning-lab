import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_xor(X, Y, lr=0.05, epochs=1000):

    w1 = np.random.randn(2, 2) * 0.5
    w2 = np.random.randn(2) * 0.5

    for epoch in range(epochs):

        error = 0

        for x, target in zip(X, Y):

            hidden = sigmoid(x @ w1)
            output = sigmoid(hidden @ w2)

            e = target - output

            d2 = e * output * (1 - output)

            d1 = hidden * (1 - hidden) * w2 * d2

            w2 += lr * d2 * hidden
            w1 += lr * np.outer(x, d1)

            error += e ** 2

        if error / 2 <= 0.002:
            break

    return w1, w2, epoch + 1


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([0, 1, 1, 0])

w1, w2, epochs = train_xor(X, Y)

print("Epochs:", epochs)

for x in X:

    hidden = sigmoid(x @ w1)
    output = sigmoid(hidden @ w2)

    print(x, "->", round(output, 3))