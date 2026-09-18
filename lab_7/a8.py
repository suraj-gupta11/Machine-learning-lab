# A8: Hyper-parameter Tuning using GridSearchCV

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create KNN classifier
knn = KNeighborsClassifier()

# 4. Define hyper-parameters
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# 5. Create GridSearchCV
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# 6. Perform hyper-parameter tuning
grid_search.fit(X_train, y_train)

# 7. Display best hyper-parameters
print("Best Hyper-parameters:")
print(grid_search.best_params_)

# 8. Display best cross-validation accuracy
print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)

# 9. Get the best classifier
best_model = grid_search.best_estimator_

# 10. Predict on test data
y_pred = best_model.predict(X_test)

# 11. Calculate test accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(accuracy)
