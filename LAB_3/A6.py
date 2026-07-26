import pandas as pd
from scipy.spatial.distance import minkowski

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Your Minkowski Function --------------------

def minkowski_distance(vector1, vector2, p):

    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")

    distance = 0

    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p

    return distance ** (1 / p)

# -------------------- Select Two Feature Vectors --------------------

features = ["Income", "MntWines", "MntMeatProducts"]

data = df[features].fillna(0)

vector1 = data.iloc[0].values
vector2 = data.iloc[1].values

print("Vector 1:", vector1)
print("Vector 2:", vector2)

print("\nComparison of Distances")
print("-" * 60)

# -------------------- Compare for p = 1 to 10 --------------------

for p in range(1, 11):

    my_distance = minkowski_distance(vector1, vector2, p)

    scipy_distance = minkowski(vector1, vector2, p)

    print(f"p = {p}")
    print(f"My Function      : {my_distance:.4f}")
    print(f"Scipy Function   : {scipy_distance:.4f}")
    print("-" * 60)