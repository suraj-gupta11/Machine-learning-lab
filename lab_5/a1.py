import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("cb578713-bc96-42b2-961f-4664f8097953 (1).xlsx",
                   sheet_name="thyroid0387_UCI")

# Remove Record ID
df = df.drop("Record ID", axis=1)


# -------------------------------
# 1. Encoding
# -------------------------------

def encode_data(df):

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = df[col].astype(str)

            values = df[col].unique()

            mapping = {
                value: i
                for i, value in enumerate(values)
            }

            df[col] = df[col].map(mapping)

    return df


# -------------------------------
# 2. Imputation
# -------------------------------

def impute_data(df):

    for col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(
                df[col].median()
            )

    return df


# -------------------------------
# 3. Euclidean Distance
# -------------------------------

def euclidean_distance(x1, x2):

    return np.sqrt(
        np.sum((x1 - x2) ** 2)
    )


# -------------------------------
# 4. Bubble Sort
# -------------------------------

def bubble_sort(data):

    data = data.copy()

    for i in range(len(data)):

        for j in range(len(data) - i - 1):

            if data[j][0] > data[j + 1][0]:

                data[j], data[j + 1] = \
                    data[j + 1], data[j]

    return data


# -------------------------------
# 5. Selection Sort
# -------------------------------

def selection_sort(data):

    data = data.copy()

    for i in range(len(data)):

        minimum = i

        for j in range(i + 1, len(data)):

            if data[j][0] < data[minimum][0]:
                minimum = j

        data[i], data[minimum] = \
            data[minimum], data[i]

    return data


# -------------------------------
# 6. Insertion Sort
# -------------------------------

def insertion_sort(data):

    data = data.copy()

    for i in range(1, len(data)):

        key = data[i]

        j = i - 1

        while j >= 0 and data[j][0] > key[0]:

            data[j + 1] = data[j]

            j -= 1

        data[j + 1] = key

    return data


# -------------------------------
# 7. Find k Neighbors
# -------------------------------

def find_neighbors(X_train, y_train,
                   test_point, k,
                   sorting="bubble"):

    distances = []

    for i in range(len(X_train)):

        d = euclidean_distance(
            X_train[i],
            test_point
        )

        distances.append(
            (d, y_train[i])
        )

    if sorting == "bubble":
        distances = bubble_sort(distances)

    elif sorting == "selection":
        distances = selection_sort(distances)

    elif sorting == "insertion":
        distances = insertion_sort(distances)

    return distances[:k]


# -------------------------------
# 8. Majority Voting
# -------------------------------

def majority_vote(neighbors):

    votes = {}

    for distance, label in neighbors:

        votes[label] = \
            votes.get(label, 0) + 1

    max_votes = max(votes.values())

    winners = [
        label for label, count in votes.items()
        if count == max_votes
    ]

    # Tie breaking
    if len(winners) > 1:

        return neighbors[0][1]

    return winners[0]


# -------------------------------
# Preprocess dataset
# -------------------------------

df = encode_data(df)

df = impute_data(df)

print(df.head())
print("A1 completed")