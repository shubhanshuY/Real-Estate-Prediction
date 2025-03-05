from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import joblib
from preprocessing import load_data, clean_data, feature_engineering, split_data

# Load and preprocess data
df = load_data("/Users/shubhanshu/Projects/RE1/data/data.csv")

# Clean and preprocess data
df = clean_data(df)  # Drop missing values
df = feature_engineering(df)  # Standardize features

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(df, "MEDV")

# Train the model
model = LinearRegression()

# Set up cross-validation and grid search (no hyperparameters for linear regression, but you can add some)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(model, param_grid={}, cv=kfold, scoring='r2')
grid.fit(X_train, y_train)

# Save the trained model
joblib.dump(grid.best_estimator_, "model.pkl")
print("Model saved as model.pkl")

# Evaluate the model
y_pred = grid.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"RMSE: {rmse}")

# Optionally, print R-squared value for better understanding of model fit
r2 = grid.best_estimator_.score(X_test, y_test)
print(f"R-squared: {r2}")
