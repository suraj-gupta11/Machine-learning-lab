import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train(X, Y, weights, lr=0.1, epochs=1000):

    for _ in range(epochs):

        for x, target in zip(X, Y):

            output = sigmoid(np.dot(x, weights))

            error = target - output

            weights += lr * error * output * (1 - output) * x

    return weights


# Customer data
X = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
])

Y = np.array([
    1, 1, 1, 0, 1,
    0, 1, 1, 0, 0
])

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias
X = np.c_[np.ones(len(X)), X]

weights = np.zeros(X.shape[1])

weights = train(X, Y, weights)

print("Weights:", weights)

for x in X:

    probability = sigmoid(np.dot(x, weights))
    prediction = 1 if probability >= 0.5 else 0

    print("Probability:", round(probability, 3),
          "Class:", prediction)