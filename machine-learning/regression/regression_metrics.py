from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

actual = np.array([10, 15, 12, 14, 18])
predicted = np.array([12, 8, 10, 16, 17])

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predicted)

print("\nRegression Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
