import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path = "Lab Session Data (1).xlsx"

excel_file = pd.ExcelFile(file_path)

print(excel_file.sheet_names)
df = pd.read_excel(file_path, sheet_name="marketing_campaign")
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)

# ---------------- A1 ----------------
# Study the features in the marketing_campaign dataset

print("\n========== DATASET INFORMATION ==========\n")

# Total rows and columns
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n========== FEATURE ANALYSIS ==========\n")

for column in df.columns:
    print(f"Feature Name : {column}")
    print(f"Data Type    : {df[column].dtype}")
    print(f"Missing Values : {df[column].isnull().sum()}")

    # Show unique values for categorical columns
    if df[column].dtype == 'object':
        print("Unique Values :")
        print(df[column].unique())
    else:
        print(f"Minimum Value : {df[column].min()}")
        print(f"Maximum Value : {df[column].max()}")

    print("-" * 60)