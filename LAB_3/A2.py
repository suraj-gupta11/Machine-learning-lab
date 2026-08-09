import pandas as pd
import numpy as np

# Load dataset
file_path = "Lab Session Data (1).xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")


# Function
def label_encode(column):
    unique_values = column.unique()

    mapping = {}

    for index, value in enumerate(unique_values):
        mapping[value] = index

    encoded_column = column.map(mapping)

    return encoded_column, mapping


# Call function
encoded_education, education_mapping = label_encode(df["Education"])

print(encoded_education.head())
print(education_mapping)

# Function for One-Hot Encoding

def one_hot_encode(column):
    """
    Performs One-Hot Encoding on a pandas Series.

    Parameters:
        column : pandas Series

    Returns:
        encoded_dataframe : pandas DataFrame
    """

    unique_values = column.unique()

    encoded_df = pd.DataFrame()

    for value in unique_values:
        encoded_df[str(value)] = (column == value).astype(int)

    return encoded_df

encoded_marital = one_hot_encode(df["Marital_Status"])

print("\nOne-Hot Encoded Marital Status")
print(encoded_marital.head())