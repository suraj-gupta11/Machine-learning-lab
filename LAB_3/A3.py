import pandas as pd

# -------------------- Load Dataset --------------------

file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# -------------------- Label Encoding Function --------------------

def label_encode(column):
    unique_values = column.unique()

    mapping = {}

    for index, value in enumerate(unique_values):
        mapping[value] = index

    encoded_column = column.map(mapping)

    return encoded_column, mapping

# -------------------- One-Hot Encoding Function --------------------

def one_hot_encode(column):
    unique_values = column.unique()

    encoded_df = pd.DataFrame()

    for value in unique_values:
        encoded_df[str(value)] = (column == value).astype(int)

    return encoded_df

# -------------------- Find Categorical Features --------------------

categorical_columns = df.select_dtypes(include=['object']).columns

print("Categorical Features:")
print(list(categorical_columns))

# ======================================================
# Label Encoding
# ======================================================

label_encoded_df = df.copy()
label_mappings = {}

for column in categorical_columns:
    label_encoded_df[column], label_mappings[column] = label_encode(df[column])

print("\n========== LABEL ENCODED DATASET ==========")
print(label_encoded_df.head())

print("\nFeature Dimensionality after Label Encoding:")
print(label_encoded_df.shape)

print("\nLabel Encoding Mappings:")

for column, mapping in label_mappings.items():
    print(f"\n{column}")
    print(mapping)

# ======================================================
# One-Hot Encoding
# ======================================================

one_hot_df = df.copy()

for column in categorical_columns:

    encoded = one_hot_encode(one_hot_df[column])

    one_hot_df = one_hot_df.drop(column, axis=1)

    one_hot_df = pd.concat([one_hot_df, encoded], axis=1)

print("\n========== ONE-HOT ENCODED DATASET ==========")
print(one_hot_df.head())

print("\nFeature Dimensionality after One-Hot Encoding:")
print(one_hot_df.shape)

# ======================================================
# Comparison
# ======================================================

print("\n========== DIMENSION COMPARISON ==========")
print("Original Dataset      :", df.shape)
print("Label Encoded Dataset :", label_encoded_df.shape)
print("One-Hot Encoded Dataset:", one_hot_df.shape)