# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load Dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Basic Data Cleaning
data = df[["Hours_Studied", "Exam_Score"]].dropna()

# Data Visualization
plt.scatter(data["Hours_Studied"], data["Exam_Score"], color='blue')
plt.title("Study Hours vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()

# Split into Training & Testing Sets
X = data[["Hours_Studied"]]  # Features
y = data["Exam_Score"]       # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization of Predictions
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.scatter(X_test, y_pred, color='red', label="Predicted")
plt.title("Actual vs Predicted Exam Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()

# Polynomial Regression (Degree 2)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

# Predict with polynomial model
y_poly_pred = poly_model.predict(poly.transform(X_test))

print("Polynomial MSE:", mean_squared_error(y_test, y_poly_pred))
print("Polynomial R² Score:", r2_score(y_test, y_poly_pred))
