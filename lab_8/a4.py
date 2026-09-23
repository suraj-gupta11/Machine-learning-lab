import numpy as np
import matplotlib.pyplot as plt


def step(y):
    return 1 if y >= 0 else 0


def train(X, Y, weights, lr):
    for epoch in range(1000):

        error = 0

        for x, target in zip(X, Y):
            output = step(np.dot(x, weights))
            e = target - output

            weights = weights + lr * e * x
            error += e ** 2

        if error / 2 <= 0.002:
            return epoch + 1

    return 1000


X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

Y = np.array([0, 0, 0, 1])

learning_rates = np.arange(0.1, 1.1, 0.1)
epochs = []

for lr in learning_rates:

    w = np.array([10.0, 0.2, -0.75])

    e = train(X, Y, w, lr)
    epochs.append(e)

    print("Learning rate:", round(lr, 1),
          "Epochs:", e)


plt.plot(learning_rates, epochs, marker="o")
plt.xlabel("Learning Rate")
plt.ylabel("Iterations")
plt.title("Learning Rate vs Iterations")
plt.show()