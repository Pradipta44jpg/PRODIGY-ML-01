# ============================================
# House Price Prediction - Advanced Version
# Author: Pradipta Saha
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("train.csv")

# ===============================
# 2. Select Important Features
# ===============================
features = [
    'GrLivArea',      # Living area (sq ft)
    'BedroomAbvGr',   # Bedrooms
    'FullBath',       # Bathrooms
    'OverallQual',    # Overall quality
    'GarageCars',     # Garage capacity
    'YearBuilt'       # Year built
]

X = data[features]
y = data['SalePrice']

# ===============================
# 3. Handle Missing Values
# ===============================
X.fillna(X.median(), inplace=True)

# ===============================
# 4. Feature Scaling
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 5. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# 6. Initialize Models
# ===============================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    )
}

# ===============================
# 7. Train & Evaluate Models
# ===============================
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = (model, mse, r2)

    print(f"\n{name}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")

# ===============================
# 8. Choose Best Model
# ===============================
best_model_name = max(results, key=lambda x: results[x][2])
best_model = results[best_model_name][0]

print("\nBest Model:", best_model_name)

# ===============================
# 9. Custom Prediction
# ===============================
# Example house
# 2000 sq ft, 3 beds, 2 baths, quality 7, garage 2, built 2015
new_house = np.array([[2000, 3, 2, 7, 2, 2015]])
new_house_scaled = scaler.transform(new_house)

predicted_price = best_model.predict(new_house_scaled)
print(f"\nPredicted House Price: ₹{predicted_price[0]:,.2f}")

# ===============================
# 10. Visualization
# ===============================
best_preds = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_preds, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title(f"Actual vs Predicted Prices ({best_model_name})")
plt.grid(True)
plt.show()
