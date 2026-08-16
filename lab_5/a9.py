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
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Condition", axis=1)
y = df["Condition"]


# ==========================================
# 4. ENCODE CATEGORICAL DATA
# ==========================================

X = pd.get_dummies(X)


# ==========================================
# 5. HANDLE MISSING VALUES
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


# Convert to NumPy
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# ==========================================
# 7. WEIGHTED kNN
# ==========================================

class WeightedKNN:

    def __init__(self, k=3):
        self.k = k


    # --------------------------------------
    # Fit()
    # --------------------------------------

    def Fit(self, X, y):

        self.X_train = X
        self.y_train = y

        return self


    # --------------------------------------
    # Predict()
    # --------------------------------------

    def Predict(self, X):

        predictions = []

        for test_point in X:

            # Calculate Euclidean distance
            distances = np.sqrt(
                np.sum(
                    (self.X_train - test_point) ** 2,
                    axis=1
                )
            )

            # Find k nearest neighbors
            nearest = np.argpartition(
                distances,
                self.k - 1
            )[:self.k]

            labels = self.y_train[nearest]
            nearest_distances = distances[nearest]

            # Calculate weighted votes
            votes = {}

            for distance, label in zip(
                nearest_distances,
                labels
            ):

                # Higher weight for closer points
                weight = 1 / (distance + 0.000001)

                votes[label] = \
                    votes.get(label, 0) + weight

            # Select class with highest weight
            prediction = max(
                votes,
                key=votes.get
            )

            predictions.append(prediction)

        return np.array(predictions)


    # --------------------------------------
    # Score()
    # --------------------------------------

    def Score(self, X, y):

        predictions = self.Predict(X)

        return np.mean(
            predictions == y
        )


# ==========================================
# 8. TEST DIFFERENT VALUES OF K
# ==========================================

k_values = range(1, 11)

weighted_accuracy = []
normal_accuracy = []
sklearn_accuracy = []


for k in k_values:

    # ======================================
    # Weighted kNN
    # ======================================

    weighted_model = WeightedKNN(k)

    weighted_model.Fit(
        X_train,
        y_train
    )

    accuracy1 = weighted_model.Score(
        X_test,
        y_test
    )

    weighted_accuracy.append(
        accuracy1
    )


    # ======================================
    # Normal kNN - A8 comparison
    # ======================================

    normal_model = KNeighborsClassifier(
        n_neighbors=k
    )

    normal_model.fit(
        X_train,
        y_train
    )

    accuracy2 = normal_model.score(
        X_test,
        y_test
    )

    normal_accuracy.append(
        accuracy2
    )


# ==========================================
# 9. PRINT RESULTS
# ==========================================

print("==========================================")
print("       A9 WEIGHTED kNN RESULTS")
print("==========================================")

for k, weighted, normal in zip(
    k_values,
    weighted_accuracy,
    normal_accuracy
):

    print(
        "k =", k,
        "| Weighted kNN =",
        round(weighted, 4),
        "| Normal kNN =",
        round(normal, 4)
    )


# ==========================================
# 10. PLOT COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    weighted_accuracy,
    marker="o",
    label="Weighted kNN"
)

plt.plot(
    list(k_values),
    normal_accuracy,
    marker="s",
    label="Normal sklearn kNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")

plt.title(
    "Weighted kNN vs Normal kNN"
)

plt.xticks(
    list(k_values)
)

plt.legend()

plt.grid()

plt.show()