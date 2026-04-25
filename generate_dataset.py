import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

# -----------------------------
# REAL FUEL CONSUMPTION RANGE (km/L)
# -----------------------------
fuel_efficiency_range = {
    110: (50, 70),
    125: (45, 55),
    150: (35, 60),
    155: (40, 48),
    160: (40, 50),
    200: (30, 45)
}

# -----------------------------
# REAL TOP SPEED RANGE (km/h)
# -----------------------------
top_speed_range = {
    110: (80, 95),
    125: (96, 112),
    150: (90, 120),
    155: (111, 122),
    160: (110, 135),
    200: (115, 140)
}

cc_list = [110, 125, 150, 155, 160, 200]

rows = []

for i in range(n):

    distance = np.random.randint(5, 120)
    load = np.random.randint(60, 250)
    engine_cc = np.random.choice(cc_list)

    # -----------------------------
    # BIKE PHYSICS PARAMETERS
    # -----------------------------
    min_ts, max_ts = top_speed_range[engine_cc]
    bike_top_speed = np.random.uniform(min_ts, max_ts)

    min_fe, max_fe = fuel_efficiency_range[engine_cc]
    base_efficiency = np.random.uniform(min_fe, max_fe)

    time_steps = np.random.randint(30, 120)

    speed_series = []
    current_speed = np.random.randint(20, 50)

    for t in range(time_steps):

        change = np.random.normal(0, 6)

        # cruising behavior
        if current_speed < bike_top_speed * 0.6:
            change += 2
        elif current_speed > bike_top_speed * 0.85:
            change -= 3

        current_speed += change
        current_speed = max(0, min(current_speed, bike_top_speed))

        if np.random.rand() < 0.04:
            current_speed = np.random.randint(0, 8)

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
    # PHYSICS-BASED FUEL MODEL
    # -----------------------------

    optimal_speed = bike_top_speed * 0.6

    # speed efficiency curve
    speed_deviation = abs(avg_speed - optimal_speed) / bike_top_speed
    speed_factor = np.exp(-speed_deviation * 2)

    # nonlinear drag (v².5 effect)
    drag_ratio = avg_speed / bike_top_speed
    drag_penalty = (drag_ratio ** 2.5) * 10

    # stop-go + stress
    idle_penalty = idle_time * 0.02
    accel_penalty = accel_intensity * 0.04
    load_penalty = max(0, (load - 100) * 0.012)

    # final efficiency
    efficiency = base_efficiency * (0.55 + 0.45 * speed_factor)

    efficiency -= drag_penalty
    efficiency -= idle_penalty
    efficiency -= accel_penalty
    efficiency -= load_penalty

    efficiency = np.clip(efficiency, 8, 80)

    fuel_used = distance / efficiency

    # -----------------------------
    # STORE ROW
    # -----------------------------
    rows.append({
        "distance": distance,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "speed_std": speed_std,
        "idle_time": idle_time,
        "accel_intensity": accel_intensity,
        "load": load,
        "engine_cc": engine_cc,
        "bike_top_speed": bike_top_speed,
        "fuel_efficiency_km_l": efficiency,
        "fuel_used": fuel_used
    })

# -----------------------------
# DATAFRAME OUTPUT
# -----------------------------
df = pd.DataFrame(rows)
df.to_csv("motorcycle_physics_dataset.csv", index=False)

print("✅ Physics-based dataset created!")
print(df.head())

print("\n📊 Summary:")
print(df.describe())