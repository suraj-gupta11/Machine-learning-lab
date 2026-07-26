import pandas as pd
import numpy as np

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Select Two Feature Vectors --------------------

features = ["Income", "MntWines", "MntMeatProducts"]

data = df[features].fillna(0)

A = data.iloc[0].values
B = data.iloc[1].values

print("Vector A:", A)
print("Vector B:", B)

# -------------------- Dot Product Function --------------------

def dot_product(A, B):

    if len(A) != len(B):
        raise ValueError("Vectors must have the same length.")

    result = 0

    for i in range(len(A)):
        result += A[i] * B[i]

    return result

# -------------------- Euclidean Norm Function --------------------

def euclidean_norm(vector):

    sum_of_squares = 0

    for value in vector:
        sum_of_squares += value ** 2

    return sum_of_squares ** 0.5

# -------------------- Using Own Functions --------------------

my_dot = dot_product(A, B)

my_norm_A = euclidean_norm(A)
my_norm_B = euclidean_norm(B)

# -------------------- Using NumPy --------------------

numpy_dot = np.dot(A, B)

numpy_norm_A = np.linalg.norm(A)
numpy_norm_B = np.linalg.norm(B)

# -------------------- Display Results --------------------

print("\n========== DOT PRODUCT ==========")
print("My Function :", my_dot)
print("NumPy       :", numpy_dot)

print("\n========== VECTOR LENGTH (EUCLIDEAN NORM) ==========")
print("Vector A")
print("My Function :", my_norm_A)
print("NumPy       :", numpy_norm_A)

print("\nVector B")
print("My Function :", my_norm_B)
print("NumPy       :", numpy_norm_B)