import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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


# Convert everything to numbers
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
# 7. OUR OWN kNN CLASSIFIER
# ==========================================

class MyKNN:

    def __init__(self, k=3):
        self.k = k


    # ======================================
    # Fit()
    # ======================================

    def Fit(self, X, y):

        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        return self


    # ======================================
    # Predict()
    # ======================================

    def Predict(self, X):

        X = np.asarray(X, dtype=float)

        predictions = []

        for test_point in X:

            # Calculate distance from test point
            distances = np.sqrt(
                np.sum(
                    (self.X_train - test_point) ** 2,
                    axis=1
                )
            )

            # Get indexes of k nearest points
            nearest_indexes = np.argsort(
                distances
            )[:self.k]

            # Get their classes
            nearest_labels = self.y_train[
                nearest_indexes
            ]

            # Majority voting
            labels, counts = np.unique(
                nearest_labels,
                return_counts=True
            )

            prediction = labels[
                np.argmax(counts)
            ]

            predictions.append(prediction)

        return np.array(predictions)


    # ======================================
    # Score()
    # ======================================

    def Score(self, X, y):

        predictions = self.Predict(X)

        accuracy = np.mean(
            predictions == np.asarray(y)
        )

        return accuracy


# ==========================================
# 8. CREATE MODEL
# ==========================================

model = MyKNN(k=3)


# ==========================================
# 9. FIT MODEL
# ==========================================

model.Fit(
    X_train,
    y_train
)


# ==========================================
# 10. PREDICT
# ==========================================

predictions = model.Predict(
    X_test
)


# ==========================================
# 11. SCORE
# ==========================================

accuracy = model.Score(
    X_test,
    y_test
)


# ==========================================
# 12. OUTPUT
# ==========================================

print("Classes:", classes)

print("\nNumber of training samples:",
      len(X_train))

print("Number of testing samples:",
      len(X_test))

print("\nFirst 20 predictions:")
print(predictions[:20])

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy (%):")
print(accuracy * 100)