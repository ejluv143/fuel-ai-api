import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("gps_mixed_fleet_dataset.csv")
df = df.dropna()

# -----------------------------
# FEATURE ENGINEERING (SAFE + CONSISTENT)
# -----------------------------

df["speed_ratio"] = df["speed"] / df["engine_cc"]
df["distance_per_sec"] = df["distance"] / 1  # already cumulative per step
df["log_speed"] = np.log1p(df["speed"])
df["load_dummy"] = 1  # placeholder (no load in this dataset)

# -----------------------------
# FEATURES & TARGET
# -----------------------------
features = [
    "speed",
    "engine_cc",
    "distance",
    "speed_ratio",
    "log_speed"
]

X = df[features].values
y = df["fuel_rate"].values   # ✅ FIXED TARGET

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# -----------------------------
# COST FUNCTION
# -----------------------------
def compute_cost(y_true, y_pred):
    m = len(y_true)
    return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

# -----------------------------
# TRACK BEST MODEL
# -----------------------------
best_model = None
best_name = ""
best_cost = float("inf")
best_transform = None

results = []

# =========================================================
# 1. LINEAR REGRESSION
# =========================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cv_s = scaler.transform(X_cv)

lr = LinearRegression()
lr.fit(X_train_s, y_train)

cv_pred = lr.predict(X_cv_s)
results.append(("LinearRegression", compute_cost(y_cv, cv_pred)))

if compute_cost(y_cv, cv_pred) < best_cost:
    best_cost = compute_cost(y_cv, cv_pred)
    best_model = lr
    best_name = "LinearRegression"
    best_transform = ("scaler", scaler)

# =========================================================
# 2. RIDGE
# =========================================================
for alpha in [0.1, 1, 10]:

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cv_s = scaler.transform(X_cv)

    model = Ridge(alpha=alpha)
    model.fit(X_train_s, y_train)

    cv_pred = model.predict(X_cv_s)

    results.append((f"Ridge_{alpha}", compute_cost(y_cv, cv_pred)))

    if compute_cost(y_cv, cv_pred) < best_cost:
        best_cost = compute_cost(y_cv, cv_pred)
        best_model = model
        best_name = f"Ridge_{alpha}"
        best_transform = ("scaler", scaler)

# =========================================================
# 3. POLYNOMIAL (optional but useful)
# =========================================================
for degree in [2, 3]:

    poly = PolynomialFeatures(degree, include_bias=False)
    scaler = StandardScaler()

    X_train_p = scaler.fit_transform(poly.fit_transform(X_train))
    X_cv_p = scaler.transform(poly.transform(X_cv))

    model = LinearRegression()
    model.fit(X_train_p, y_train)

    cv_pred = model.predict(X_cv_p)

    results.append((f"Poly_{degree}", compute_cost(y_cv, cv_pred)))

    if compute_cost(y_cv, cv_pred) < best_cost:
        best_cost = compute_cost(y_cv, cv_pred)
        best_model = model
        best_name = f"Poly_{degree}"
        best_transform = ("poly_scaler", poly, scaler)

# -----------------------------
# RESULTS
# -----------------------------
print("\n===== MODEL RESULTS =====\n")

for r in results:
    print(r[0], "CV Cost:", r[1])

print("\nBEST MODEL:", best_name)

# -----------------------------
# TEST EVALUATION
# -----------------------------
if best_name.startswith("Poly"):

    poly = best_transform[1]
    scaler = best_transform[2]

    X_test_p = scaler.transform(poly.transform(X_test))
    test_pred = best_model.predict(X_test_p)

else:
    scaler = best_transform[1]
    X_test_s = scaler.transform(X_test)
    test_pred = best_model.predict(X_test_s)

test_cost = compute_cost(y_test, test_pred)

print("\nTEST COST:", test_cost)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump({
    "model": best_model,
    "name": best_name,
    "transform": best_transform,
    "features": features
}, "best_fuel_model.pkl")

print("\n🚀 TRAINING COMPLETE")
print("Best Model:", best_name)
print("Test Cost:", test_cost)
print("Saved: best_fuel_model.pkl")