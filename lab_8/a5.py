import numpy as np


def step(x):
    if x >= 0:
        return 1
    else:
        return 0


def train(X, Y, weights, learning_rate):

    for epoch in range(1000):

        total_error = 0

        for x, target in zip(X, Y):

            net = np.dot(x, weights)

            output = step(net)

            error = target - output

            weights = weights + learning_rate * error * x

            total_error += error ** 2

        # Show progress
        if (epoch + 1) % 100 == 0:
            print("Epoch:", epoch + 1,
                  "Error:", total_error / 2)

        if total_error / 2 <= 0.002:
            return weights, epoch + 1

    return weights, 1000


# XOR INPUT
# [Bias, A, B]

X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# XOR TARGET
Y = np.array([0, 1, 1, 0])


# Initial weights
weights = np.array([
    10.0,
    0.2,
    -0.75
])


# Learning rate
learning_rate = 0.05


print("Starting A5...")

weights, epochs = train(
    X,
    Y,
    weights,
    learning_rate
)


print("\nTraining finished")
print("Final weights:", weights)
print("Epochs:", epochs)


print("\nXOR OUTPUT")

for x in X:

    net = np.dot(x, weights)

    output = step(net)

    print(
        "A =", x[1],
        "B =", x[2],
        "Z =", output
    )