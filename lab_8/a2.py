import numpy as np
import matplotlib.pyplot as plt


def step(y):
    return 1 if y >= 0 else 0


def train_perceptron(X, Y, w, lr, max_epochs=1000):
    errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for x, target in zip(X, Y):
            y = step(np.dot(x, w))
            e = target - y

            w = w + lr * e * x
            total_error += e ** 2

        errors.append(total_error / 2)

        if errors[-1] <= 0.002:
            return w, epoch + 1, errors

    return w, max_epochs, errors


# AND data
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

Y = np.array([0, 0, 0, 1])

# W0, W1, W2
weights = np.array([10.0, 0.2, -0.75])

weights, epochs, errors = train_perceptron(
    X, Y, weights, 0.05
)

print("Final weights:", weights)
print("Epochs:", epochs)

plt.plot(range(1, epochs + 1), errors)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("AND Gate - Error vs Epoch")
plt.show()