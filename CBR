# random_forest_cbr_prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Sample data preparation

data = {
    'Fiber_Content': [0.25, 0.5, 0.75, 1.0, 1.25, 0.25, 0.5, 0.75, 1.0],  # % fiber
    'Fiber_Type': [0, 0, 0, 0, 0, 1, 1, 1, 1],  # Stem=0, Root=1
    'Dry_Density': [1.525, 1.508, 1.510, 1.534, 1.532, 1.47, 1.49, 1.49, 1.48],
    'Moisture_Content': [18.48, 16.21, 16.90, 16.21, 16.09, 15.27, 14.23, 12.79, 14.06],
    'CBR_Value': [2.47, 2.94, 3.67, 5.02, 4.16, 3.14, 3.64, 5.28, 3.78]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['Fiber_Content', 'Fiber_Type', 'Dry_Density', 'Moisture_Content']]
y = df['CBR_Value']

# Split data into train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Model Performance Metrics:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Predict on entire dataset for plotting actual vs predicted
y_pred_all = rf.predict(X)

# Plot Actual vs Predicted CBR Values
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred_all, color='blue', label='Data points')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal fit (y=x)')
plt.xlabel('Actual CBR Value (%)')
plt.ylabel('Predicted CBR Value (%)')
plt.title('Actual vs Predicted CBR Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_cbr.png")
plt.show()

# Feature Importance Plot
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices], color='green', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Plot CBR vs Fiber Content for each fiber type
plt.figure(figsize=(8,6))
for fiber_type in df['Fiber_Type'].unique():
    subset = df[df['Fiber_Type'] == fiber_type]
    label = 'Stem Fiber' if fiber_type == 0 else 'Root Fiber'
    plt.plot(subset['Fiber_Content'], subset['CBR_Value'], marker='o', label=label)

plt.xlabel('Fiber Content (%)')
plt.ylabel('CBR Value (%)')
plt.title('CBR Value vs Fiber Content by Fiber Type')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cbr_vs_fiber_content.png")
plt.show()
