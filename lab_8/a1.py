import numpy as np


# ---------- Summation ----------
def summation(x, w, bias):
    return np.dot(x, w) + bias


# ---------- Activation Functions ----------
def step(y):
    return 1 if y >= 0 else 0


def bipolar_step(y):
    return 1 if y >= 0 else -1


def sigmoid(y):
    return 1 / (1 + np.exp(-y))


def tanh(y):
    return np.tanh(y)


def relu(y):
    return max(0, y)


def leaky_relu(y):
    return y if y > 0 else 0.01 * y


# ---------- Comparator ----------
def error(target, output):
    return target - output


# ---------- Main ----------
x = [1, 2]
w = [0.5, 0.5]
b = 1

y = summation(x, w, b)

print("Summation:", y)
print("Step:", step(y))
print("Bipolar Step:", bipolar_step(y))
print("Sigmoid:", sigmoid(y))
print("TanH:", tanh(y))
print("ReLU:", relu(y))
print("Leaky ReLU:", leaky_relu(y))
print("Error:", error(1, step(y)))