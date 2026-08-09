import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Select Feature --------------------

feature = "Income"

# Remove missing values
data = df[feature].dropna()

# -------------------- Mean and Variance --------------------

mean = np.mean(data)
variance = np.var(data)

print("Feature :", feature)
print("Mean :", mean)
print("Variance :", variance)

# -------------------- Histogram Data --------------------

hist, bins = np.histogram(data, bins=10)

print("\nHistogram Counts:")
print(hist)

print("\nBin Edges:")
print(bins)

# -------------------- Plot Histogram --------------------

plt.figure(figsize=(8,5))

plt.hist(data, bins=10, edgecolor='black')

plt.title(f"Histogram of {feature}")
plt.xlabel(feature)
plt.ylabel("Frequency")

plt.grid(True)

plt.show()