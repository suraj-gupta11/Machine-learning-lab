from sklearn.neural_network import MLPClassifier


X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y = [0, 0, 0, 1]

model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="logistic",
    learning_rate_init=0.05,
    max_iter=1000,
    random_state=1
)

model.fit(X, Y)

print("AND predictions:")
print(model.predict(X))

from sklearn.neural_network import MLPClassifier


X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y = [0, 1, 1, 0]

model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="logistic",
    learning_rate_init=0.05,
    max_iter=1000,
    random_state=1
)

model.fit(X, Y)

print("XOR predictions:")
print(model.predict(X))