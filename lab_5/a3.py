import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel(
    "cb578713-bc96-42b2-961f-4664f8097953 (1).xlsx",
    sheet_name="thyroid0387_UCI"
)

# Select two classes
classes = df["Condition"].dropna().unique()[:2]

df = df[df["Condition"].isin(classes)]

# Separate features and target
X = df.drop("Condition", axis=1)
y = df["Condition"]

# Convert categorical columns to numbers
X = pd.get_dummies(X)

# Fill missing values
X = X.fillna(X.median(numeric_only=True))

# Fill any remaining missing values
X = X.fillna(0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Classes:", classes)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)