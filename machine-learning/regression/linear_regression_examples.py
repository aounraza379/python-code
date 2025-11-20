# Linear Regression Examples
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Example 1: Hours vs Score ---
data = {"Hours": [2, 4, 6, 8, 10], "Score": [45, 50, 60, 75, 95]}
df = pd.DataFrame(data)
X = df[["Hours"]]
y = df["Score"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Predicted Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.show()

# --- Example 2: Ice Cream Shop Sales ---
data = {'Temperature': [25, 28, 30, 32, 35, 38],
        'IceCreams': [50, 65, 80, 95, 110, 125]}
df = pd.DataFrame(data)
X = df[['Temperature']]
y = df['IceCreams']
model = LinearRegression()
model.fit(X, y)

predicted_icecreams = model.predict([[40]])
print(f"At 40°C, we'll sell approximately {int(predicted_icecreams[0])} ice creams!")

temperatures = [20, 25, 30, 35, 40, 45]
for temp in temperatures:
    sales = model.predict([[temp]])
    print(f"{temp}°C → {int(sales[0])} ice creams")

# --- Example 3: House Price Prediction ---
data = {'Size_sqft': [1000, 1500, 2000, 2500, 3000],
        'Price_k': [200, 280, 350, 420, 510]}
df = pd.DataFrame(data)
X = df[["Size_sqft"]]
y = df["Price_k"]

model = LinearRegression()
model.fit(X, y)

new_X = np.array([[1800], [1000]])
preds = model.predict(new_X)
print(f"Predicted Price for 1800 sqft: ${preds[0]:.2f}K")
print(f"Predicted Price for 1000 sqft: ${preds[1]:.2f}K")

print(f"Coefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")

# --- Example 4: Real Estate Dataset ---
np.random.seed(42)
n = 200
house_age = np.random.randint(1, 50, n)
distance_to_station = np.random.uniform(100, 5000, n)
num_stores = np.random.randint(0, 10, n)
price_per_unit_area = (5000 - 20*house_age - 0.5*distance_to_station + 200*num_stores +
                       np.random.normal(0, 500, n))

data = pd.DataFrame({
    'house age': house_age,
    'distance to mrt': distance_to_station,
    'number of stores': num_stores,
    'price per unit area': price_per_unit_area
})
print("Real Data Set\n", data.head())

X = data[['house age', 'distance to mrt', 'number of stores']]
y = data['price per unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

plt.scatter(y_test, y_pred, color='red', alpha=0.8)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Price per Unit Area')
plt.ylabel('Predicted Price per Unit Area')
plt.title('Actual vs Predicted Prices')
plt.show()

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())
