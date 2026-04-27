from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_fuel_model.pkl")

data = joblib.load(MODEL_PATH)

model = data["model"]
model_name = data.get("name", "LinearRegression")
transform = data.get("transform", None)
feature_names = data["features"]

# -----------------------------
# SAFE CONVERTER
# -----------------------------
def safe_float(v):
    try:
        return float(v)
    except:
        return 0.0

# -----------------------------
# ENGINE LIMITS
# -----------------------------
MAX_SPEED = {
    110: 95,
    125: 110,
    150: 120,
    155: 125,
    160: 135,
    200: 140
}

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API running 🚀",
        "model": model_name
    })

# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        req = request.get_json(force=True) or {}

        distance = safe_float(req.get("distance"))
        speed = safe_float(req.get("speed"))
        load = safe_float(req.get("load"))
        engine_cc = int(safe_float(req.get("engine_cc")))

        max_limit = MAX_SPEED.get(engine_cc, 120)

        # -----------------------------
        # SIMULATED SPEED PROFILE
        # -----------------------------
        time_steps = 60
        current_speed = speed
        speed_series = []

        for _ in range(time_steps):
            current_speed += np.random.randint(-5, 6)
            current_speed = max(0, min(current_speed, max_limit))

            if np.random.rand() < 0.08:
                current_speed = np.random.randint(0, 10)

            speed_series.append(current_speed)

        speed_series = np.array(speed_series)

        # -----------------------------
        # FEATURES
        # -----------------------------
        input_dict = {
            "distance": distance,
            "avg_speed": np.mean(speed_series),
            "max_speed": np.max(speed_series),
            "speed_std": np.std(speed_series),
            "idle_time": np.sum(speed_series < 5),
            "accel_intensity": np.mean(np.abs(np.diff(speed_series))) if len(speed_series) > 1 else 0,
            "load": load,
            "engine_cc": engine_cc
        }

        X = pd.DataFrame([input_dict])

        # ensure correct column order
        X = X[feature_names]

        # -----------------------------
        # APPLY TRANSFORMATION SAFELY
        # -----------------------------
        if transform is not None:

            if isinstance(transform, dict):
                scaler = transform.get("scaler", None)
                poly = transform.get("poly", None)

                if poly:
                    X = poly.transform(X)

                if scaler:
                    X = scaler.transform(X)

            else:
                # fallback (old format list/tuple)
                if len(transform) == 2:
                    scaler = transform[1]
                    X = scaler.transform(X)

                elif len(transform) == 3:
                    poly = transform[1]
                    scaler = transform[2]

                    X = poly.transform(X)
                    X = scaler.transform(X)

        # -----------------------------
        # PREDICTION
        # -----------------------------
        prediction = model.predict(X)[0]

        return jsonify({
            "model": model_name,
            "predicted_fuel": round(float(prediction), 3)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)