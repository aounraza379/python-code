from sklearn.linear_model import LinearRegression
import pandas as pd

# Sample data
data = {'Day': [1, 2, 3, 4, 5, 6],
        'Temp': [31, 32, 33, 34, 35, 36]}
df = pd.DataFrame(data)

# Prepare features and labels
X = df[['Day']]
y = df['Temp']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Make prediction
d = 8
predicted = model.predict([[d]])
print(f"\nPredicted Temp on Day {d} is: {predicted}")

# -----------------------------
# Example 2: Predict Score vs Hours Studied
# -----------------------------
data = {
    "Hours": [2,4,6,8,10],
    "Score": [45,50,60,75,95]
}
df = pd.DataFrame(data)
X = df[["Hours"]]
y = df["Score"]

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R score:", r2)

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Predicted Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.show()

# -----------------------------
# Example 3: Ice Cream Sales Prediction
# -----------------------------
data = {'Temperature': [25, 28, 30, 32, 35, 38],
        'IceCreams': [50, 65, 80, 95, 110, 125]}
df = pd.DataFrame(data)

X = df[['Temperature']]
y = df['IceCreams']
model = LinearRegression()
model.fit(X, y)

predicted_icecreams = model.predict([[40]])
print(f"At 40°C, we'll sell approximately {int(predicted_icecreams[0])} ice creams!")

# Forecast
temperatures = [20, 25, 30, 35, 40, 45]
for temp in temperatures:
    sales = model.predict([[temp]])
    print(f"{temp}°C →  {int(sales[0])} ice creams")

# -----------------------------
# Example 4: House Price Prediction
# -----------------------------
data = {'Size_sqft': [1000, 1500, 2000, 2500, 3000],
        'Price_k': [200, 280, 350, 420, 510]}
df = pd.DataFrame(data)

X = df[["Size_sqft"]]
y = df["Price_k"]

model = LinearRegression()
model.fit(X, y)

new_X = np.array([[1800], [1000]])
predicted_prices = model.predict(new_X)
print(f"Predicted Price for 1800 sqft: ${predicted_prices[0]:.2f}K")
print(f"Predicted Price for 1000 sqft: ${predicted_prices[1]:.2f}K")

coefficient = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {coefficient:.4f}, Intercept: {intercept:.4f}")

# -----------------------------
# Example 5: Train-Test Split Evaluation
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_final = LinearRegression()
model_final.fit(X_train, y_train)

predictions_test = model_final.predict(X_test)
mae = mean_absolute_error(y_test, predictions_test)
mse = mean_squared_error(y_test, predictions_test)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}")
