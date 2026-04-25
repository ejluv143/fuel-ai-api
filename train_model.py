import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("motorcycle_data_level3.csv")

# -----------------------------
# CLEAN DATA (IMPORTANT FIX)
# -----------------------------
df = df.dropna()

# -----------------------------
# FEATURES & TARGET
# -----------------------------
feature_cols = [
    "distance",
    "avg_speed",
    "max_speed",
    "speed_std",
    "idle_time",
    "accel_intensity",
    "load",
    "engine_cc"
]

X = df[feature_cols]
y = df["fuel_used"]

# -----------------------------
# BASELINE MODEL (for comparison)
# -----------------------------
baseline_pred = np.mean(y)
baseline_error = np.mean(np.abs(y - baseline_pred))

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# PREDICT
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# METRICS
# -----------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# -----------------------------
# RESULTS
# -----------------------------
print("\n===== MODEL EVALUATION =====")
print(f"R² Score        : {r2:.4f}")
print(f"MAE             : {mae:.4f} L")
print(f"MSE Loss        : {mse:.6f}")
print(f"RMSE            : {rmse:.4f} L")
print(f"Baseline MAE    : {baseline_error:.4f} L")

# -----------------------------
# TRAINING STATUS
# -----------------------------
if r2 >= 0.5:
    print("\n✅ Model successfully trained!")
    print("Model is ready for deployment.")

    joblib.dump({
        "model": model,
        "features": feature_cols
    }, "fuel_efficiency_model.pkl")

    print("📦 Model saved as fuel_efficiency_model.pkl")

else:
    print("\n❌ Model performance is weak.")
    print("Improve dataset or feature engineering.")