# Polynomial Regression Examples
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Example 1: Simple polynomial regression ---
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([3,6,7,8,11,12,14,16,17,19])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X, y, color='blue', label='Original data')
plt.scatter(X_test[:,1], y_pred, color='red', label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.show()

# --- Example 2: Salary Prediction based on Years of Experience ---
X = np.array([0,1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([5,7,14,28,45,68,97,130,170,215,265])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='green', label='Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Polynomial Regression: Actual vs Predicted')
plt.legend()
plt.show()
