import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

# Convert categorical features into numbers
X = pd.get_dummies(X)

# Handle missing values
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Train kNN with k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# A5: Calculate accuracy
accuracy = knn.score(X_test, y_test)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)