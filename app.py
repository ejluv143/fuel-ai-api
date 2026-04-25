from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# LOAD MODEL (FIXED FOR RENDER)
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "fuel_efficiency_model.pkl")

data = joblib.load(MODEL_PATH)
model = data["model"]
feature_names = data["features"]

# -----------------------------
# SAFE CONVERTER
# -----------------------------
def safe_float(v):
    try:
        return float(v) if v is not None else 0.0
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
    return jsonify({"status": "API running 🚀"})

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
        # SIMULATED PROFILE
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
        avg_speed = np.mean(speed_series)
        max_speed = np.max(speed_series)
        speed_std = np.std(speed_series)
        idle_time = np.sum(speed_series < 5)

        accel = np.diff(speed_series)
        accel_intensity = np.mean(np.abs(accel)) if len(accel) > 0 else 0

        # -----------------------------
        # MODEL INPUT
        # -----------------------------
        input_dict = {
            "distance": distance,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "speed_std": speed_std,
            "idle_time": idle_time,
            "accel_intensity": accel_intensity,
            "load": load,
            "engine_cc": engine_cc
        }

        features = pd.DataFrame([input_dict])[feature_names]

        prediction = model.predict(features)[0]

        return jsonify({
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