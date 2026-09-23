import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# Customer data from A6
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
], dtype=float)


# 1 = High Value
# 0 = Low Value
Y = np.array([
    1, 1, 1, 0, 1,
    0, 1, 1, 0, 0
])


# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)


# MLP model
model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation="logistic",
    learning_rate_init=0.05,
    max_iter=1000,
    random_state=1
)


# Train
model.fit(X, Y)


# Predict
prediction = model.predict(X)


print("Actual :", Y)
print("Output :", prediction)
print("Accuracy:", model.score(X, Y))