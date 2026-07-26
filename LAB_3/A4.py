import pandas as pd

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Minkowski Distance Function --------------------

def minkowski_distance(vector1, vector2, p):
    """
    Calculate the generalized Minkowski Distance.

    Parameters:
        vector1 : First vector
        vector2 : Second vector
        p       : Order of the distance

    Returns:
        Distance
    """

    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")

    distance = 0

    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p

    distance = distance ** (1 / p)

    return distance

# -------------------- Select Two Feature Vectors --------------------

features = ["Income", "MntWines", "MntMeatProducts"]

# Fill missing values if any
data = df[features].fillna(0)

vector1 = data.iloc[0].values
vector2 = data.iloc[1].values

print("Vector 1:", vector1)
print("Vector 2:", vector2)

# -------------------- Manhattan Distance --------------------

manhattan = minkowski_distance(vector1, vector2, 1)

print("\nManhattan Distance (p = 1):", manhattan)

# -------------------- Euclidean Distance --------------------

euclidean = minkowski_distance(vector1, vector2, 2)

print("Euclidean Distance (p = 2):", euclidean)