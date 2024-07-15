import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------Data Preparation----------
weather = pd.read_csv("weather.csv")

# Keep station column as strings and convert daily columns to numeric
weather.iloc[:, 3:] = weather.iloc[:, 3:].replace(r'[^\d.]+', 0, regex=True)

for col in weather.columns[3:]:
    weather[col] = pd.to_numeric(weather[col], errors='coerce')

weather.fillna(0, inplace=True)

weather['monthly_rainfall'] = weather.iloc[:, 3:34].sum(axis=1)

# Drop the daily rainfall columns
weather.drop(columns=weather.columns[3:34], inplace=True)

# Filter the data for years >= 1986
weather = weather[weather['Year'] >= 1986]

print(weather)

# One-hot encode the Station column
weather = pd.get_dummies(weather, columns=['Station'], drop_first=True)

# Split data into features and target
x = weather.drop(columns=['monthly_rainfall'])
y = weather['monthly_rainfall']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(weather)
print(f"X_train: {x_train.shape}")
print(f"X_test : {x_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")

rainfall_stats = weather['monthly_rainfall'].describe()
print("Summary Statistics for Monthly Rainfall:")
print(rainfall_stats)

# ----------Model Building----------
# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Bagging': BaggingRegressor(),
    'Extra Trees': ExtraTreeRegressor(),
}

# Evaluate models and find the best one
best_model = None
best_mse = float('inf')
best_model_name = ""

for name, model in models.items():
    try:
        scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mean_mse = -scores.mean()  # Negate to get positive MSE
        print(f"{name} with mean squared error: {mean_mse}")

        if mean_mse < best_mse:
            best_mse = mean_mse
            best_model = model
            best_model_name = name
    except Exception as e:
        print(f"Failed to evaluate {name}: {e}")

print(f"Best model: {best_model_name} with mean squared error: {best_mse}")

# Train the best model on the entire training set
best_model.fit(x_train, y_train)

# Save the best model
joblib.dump(best_model, 'rainfall_prediction_model.pkl')

# Test the best model on the test set
y_pred = best_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(f"{best_model_name} mean squared error on test set: {test_mse}")
print(f"{best_model_name} R^2 score on test set: {test_r2}")

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.xlabel('Actual Monthly Rainfall')
plt.ylabel('Predicted Monthly Rainfall')
plt.title(f'Actual vs Predicted Monthly Rainfall using {best_model_name}')
plt.show()
