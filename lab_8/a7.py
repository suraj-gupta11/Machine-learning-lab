import numpy as np


X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

Y = np.array([0, 0, 0, 1])

# Pseudo-inverse
W = np.linalg.pinv(X) @ Y

print("Pseudo-inverse weights:", W)

output = X @ W

print("Continuous output:", output)

prediction = (output >= 0.5).astype(int)

print("Prediction:", prediction)
print("Target:", Y)