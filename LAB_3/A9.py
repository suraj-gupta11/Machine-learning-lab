import pandas as pd
import numpy as np

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Select Numerical Columns --------------------

numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Replace missing values with 0 (or use dropna() if preferred)
numeric_df = numeric_df.fillna(0)

# -------------------- Own Functions --------------------

def calculate_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)

def calculate_variance(data):
    mean = calculate_mean(data)

    total = 0
    for value in data:
        total += (value - mean) ** 2

    return total / len(data)

def calculate_std(data):
    variance = calculate_variance(data)
    return variance ** 0.5

# -------------------- Compare Results --------------------

print("Feature".ljust(20),
      "My Mean".rjust(12),
      "NumPy Mean".rjust(15),
      "My Std".rjust(15),
      "NumPy Std".rjust(15))

print("-" * 80)

for column in numeric_df.columns:

    values = numeric_df[column].tolist()

    # Own Functions
    my_mean = calculate_mean(values)
    my_std = calculate_std(values)

    # NumPy Functions
    numpy_mean = np.mean(values)
    numpy_std = np.std(values)

    print(f"{column:20} "
          f"{my_mean:12.2f}"
          f"{numpy_mean:15.2f}"
          f"{my_std:15.2f}"
          f"{numpy_std:15.2f}")