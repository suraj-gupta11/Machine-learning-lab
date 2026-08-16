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

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Create kNN classifier with k = 3
knn = KNeighborsClassifier(n_neighbors=3)

# Train
knn.fit(X_train, y_train)

# A6: Predict test vectors
y_pred = knn.predict(X_test)

print("Predicted classes:")
print(y_pred)