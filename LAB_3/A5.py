import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Minkowski Distance Function --------------------

def minkowski_distance(vector1, vector2, p):

    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")

    distance = 0

    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p

    return distance ** (1 / p)

# -------------------- Select Two Feature Vectors --------------------

# Using three numerical features
features = ["Income", "MntWines", "MntMeatProducts"]

# Remove missing values
data = df[features].fillna(0)

# Select first two rows as vectors
vector1 = data.iloc[0].values
vector2 = data.iloc[1].values

print("Vector 1:", vector1)
print("Vector 2:", vector2)

# -------------------- Calculate Distance for p = 1 to 10 --------------------

p_values = []
distances = []

print("\nMinkowski Distance")

for p in range(1, 11):

    distance = minkowski_distance(vector1, vector2, p)

    p_values.append(p)
    distances.append(distance)

    print(f"p = {p} --> Distance = {distance:.2f}")

# -------------------- Plot Graph --------------------

plt.figure(figsize=(8,5))

plt.plot(p_values, distances, marker='o')

plt.title("Minkowski Distance (p = 1 to 10)")
plt.xlabel("Order (p)")
plt.ylabel("Distance")

plt.xticks(range(1,11))

plt.grid(True)

plt.show()