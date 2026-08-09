import pandas as pd

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Select Only Numerical Columns --------------------

numeric_df = df.select_dtypes(include=['int64', 'float64'])

# -------------------- Mean Function --------------------

def calculate_mean(data):

    total = 0

    for value in data:
        total += value

    return total / len(data)

# -------------------- Variance Function --------------------

def calculate_variance(data):

    mean = calculate_mean(data)

    total = 0

    for value in data:
        total += (value - mean) ** 2

    return total / len(data)

# -------------------- Standard Deviation Function --------------------

def calculate_std(data):

    variance = calculate_variance(data)

    return variance ** 0.5

# -------------------- Calculate for Each Feature --------------------

print("Feature\t\t\tMean\t\tVariance\t\tStandard Deviation")
print("-" * 90)

for column in numeric_df.columns:

    # Remove missing values
    values = numeric_df[column].dropna().tolist()

    mean = calculate_mean(values)
    variance = calculate_variance(values)
    std = calculate_std(values)

    print(f"{column:20} {mean:10.2f} {variance:15.2f} {std:15.2f}")