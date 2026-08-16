import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel(
    "cb578713-bc96-42b2-961f-4664f8097953 (1).xlsx",
    sheet_name="thyroid0387_UCI"
)


# ==========================================
# 2. SELECT TWO CLASSES
# ==========================================

classes = df["Condition"].dropna().unique()[:2]

df = df[df["Condition"].isin(classes)]


# ==========================================
# 3. SEPARATE X AND y
# ==========================================

X = df.drop("Condition", axis=1)
y = df["Condition"]


# ==========================================
# 4. ENCODING
# ==========================================

X = pd.get_dummies(X)


# ==========================================
# 5. MISSING VALUES
# ==========================================

X = X.fillna(0)

X = X.astype(float)


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ==========================================
# 7. OUR OWN kNN
# ==========================================

class MyKNN:

    def __init__(self, k=3):
        self.k = k

    def Fit(self, X, y):

        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        return self

    def Predict(self, X):

        X = np.asarray(X, dtype=float)

        predictions = []

        for test_point in X:

            # Calculate Euclidean distances
            distances = np.sqrt(
                np.sum(
                    (self.X_train - test_point) ** 2,
                    axis=1
                )
            )

            # Find k nearest neighbors
            nearest = np.argsort(distances)[:self.k]

            # Get their labels
            labels = self.y_train[nearest]

            # Majority voting
            unique, counts = np.unique(
                labels,
                return_counts=True
            )

            prediction = unique[
                np.argmax(counts)
            ]

            predictions.append(prediction)

        return np.array(predictions)

    def Score(self, X, y):

        predictions = self.Predict(X)

        return np.mean(
            predictions == np.asarray(y)
        )


# ==========================================
# 8. TEST DIFFERENT VALUES OF K
# ==========================================

k_values = range(1, 11)

my_accuracy = []
sklearn_accuracy = []


for k in k_values:

    # --------------------------------------
    # Our kNN
    # --------------------------------------

    my_model = MyKNN(k)

    my_model.Fit(
        X_train,
        y_train
    )

    accuracy1 = my_model.Score(
        X_test,
        y_test
    )

    my_accuracy.append(accuracy1)


    # --------------------------------------
    # sklearn kNN
    # --------------------------------------

    sklearn_model = KNeighborsClassifier(
        n_neighbors=k
    )

    sklearn_model.fit(
        X_train,
        y_train
    )

    accuracy2 = sklearn_model.score(
        X_test,
        y_test
    )

    sklearn_accuracy.append(accuracy2)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("Comparison of kNN accuracies")
print("--------------------------------")

for k, a1, a2 in zip(
    k_values,
    my_accuracy,
    sklearn_accuracy
):

    print(
        "k =", k,
        "| My kNN =", round(a1, 4),
        "| sklearn =", round(a2, 4)
    )


# ==========================================
# 10. PLOT ACCURACY
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    my_accuracy,
    marker="o",
    label="My kNN"
)

plt.plot(
    k_values,
    sklearn_accuracy,
    marker="s",
    label="sklearn kNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")

plt.title(
    "Comparison of My kNN and sklearn kNN"
)

plt.xticks(list(k_values))

plt.legend()

plt.grid()

plt.show()