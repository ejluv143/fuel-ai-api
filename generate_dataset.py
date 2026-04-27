import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# -----------------------------
# LOCATIONS
# -----------------------------
city_routes = [
    (14.5995, 120.9842),  # Manila
    (10.3157, 123.8854),  # Cebu
    (7.1907, 125.4553)    # Davao
]

# -----------------------------
# ENGINE CONFIG
# -----------------------------
cc_list = [110, 125, 150, 160, 200]
cc_weights = [0.4, 0.25, 0.15, 0.1, 0.1]

speed_limits = {
    110: 85,
    125: 100,
    150: 115,
    160: 125,
    200: 140
}

# -----------------------------
# TIME STEP
# -----------------------------
SEC = 5

# -----------------------------
# PHYSICS CONSTANTS
# -----------------------------
AIR_DENSITY = 1.225
DRAG_COEFF = 0.9
FRONTAL_AREA = 0.6

# -----------------------------
# REAL IDLE FUEL (L/hr)
# -----------------------------
idle_fuel_map = {
    110: (0.15, 0.30),
    125: (0.20, 0.50),
    150: (0.14, 0.25),
    160: (0.14, 0.30),
    200: (0.50, 1.20)
}

# -----------------------------
# HELPERS
# -----------------------------
def choose_trip_type():
    return np.random.choice(["city", "mixed", "long"], p=[0.5, 0.3, 0.2])

def gps_noise():
    return np.random.normal(0, 0.00002)

def sensor_noise(x, scale=0.03):
    return x + np.random.normal(0, scale * max(abs(x), 1))


# -----------------------------
# TRIP GENERATOR
# -----------------------------
def generate_trip(trip_id):

    trip_type = choose_trip_type()

    base_lat, base_lon = city_routes[np.random.randint(len(city_routes))]
    lat = base_lat + np.random.normal(0, 0.002)
    lon = base_lon + np.random.normal(0, 0.002)

    engine_cc = np.random.choice(cc_list, p=cc_weights)
    max_speed = speed_limits[engine_cc]

    speed = np.random.uniform(10, 30)
    prev_speed = speed

    bearing = np.random.uniform(0, 360)
    time = datetime.now()

    total_distance = 0
    points = []

    load = np.random.uniform(50, 150)

    # -----------------------------
    # SAFE STEP SELECTION (FIXED)
    # -----------------------------
    if trip_type == "city":
        steps = np.random.randint(80, 180)
    elif trip_type == "mixed":
        steps = np.random.randint(150, 300)
    else:
        steps = np.random.randint(300, 600)

    # -----------------------------
    # IDLE FUEL PER STEP
    # -----------------------------
    idle_low, idle_high = idle_fuel_map.get(engine_cc, (0.2, 0.4))
    idle_lph = np.random.uniform(idle_low, idle_high)
    idle_per_step = (idle_lph / 3600) * SEC

    for _ in range(steps):

        # -----------------------------
        # DRIVER BEHAVIOR
        # -----------------------------
        accel = np.random.normal(0, 3)

        if trip_type == "city":
            accel += np.random.normal(0, 2)
            stop_prob = 0.08
        elif trip_type == "mixed":
            accel += np.random.normal(0, 1)
            stop_prob = 0.04
        else:
            accel += np.random.normal(0.5, 1)
            stop_prob = 0.02

        speed += accel
        speed = np.clip(speed, 0, max_speed)

        # random stop
        if np.random.rand() < stop_prob:
            speed = np.random.uniform(0, 5)

        # -----------------------------
        # PHYSICS (DRAG)
        # -----------------------------
        speed_mps = speed / 3.6
        drag_force = 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA * speed_mps**2

        # -----------------------------
        # DISTANCE (FIXED)
        # -----------------------------
        raw_step_km = (speed * SEC) / 3600

        # apply noise
        step_km = sensor_noise(raw_step_km, 0.05)

        # 🔥 prevent fake zero movement
        if speed > 1:
            step_km = max(step_km, raw_step_km * 0.5)

        step_km = max(step_km, 0)

        total_distance += step_km

        # -----------------------------
        # GPS MOVEMENT
        # -----------------------------
        lat += (step_km * np.cos(np.radians(bearing))) / 111 + gps_noise()
        lon += (step_km * np.sin(np.radians(bearing))) / (111 * np.cos(np.radians(lat))) + gps_noise()

        bearing += np.random.normal(0, 5)
        bearing %= 360

        # -----------------------------
        # FUEL MODEL
        # -----------------------------
        accel_factor = abs(speed - prev_speed)

        fuel_rate = (
            engine_cc * 0.0003 +
            drag_force * 0.00005 +
            accel_factor * 0.002 +
            load * 0.0002
        )

        # APPLY REAL IDLE FUEL
        if speed < 5:
            fuel_rate += idle_per_step

        fuel_rate += np.random.normal(0, 0.003)
        fuel_rate = max(fuel_rate, 0.005)

        prev_speed = speed

        # -----------------------------
        # STORE
        # -----------------------------
        points.append({
            "trip_id": trip_id,
            "timestamp": time.isoformat(),
            "lat": lat,
            "lon": lon,
            "speed": speed,
            "engine_cc": engine_cc,
            "fuel_rate": fuel_rate,
            "distance": total_distance,
            "load": load,
            "trip_type": trip_type
        })

        time += timedelta(seconds=SEC)

    return points


# -----------------------------
# GENERATE DATASET
# -----------------------------
all_data = []

for t in range(500):
    all_data.extend(generate_trip(t))

df = pd.DataFrame(all_data)

df.to_csv("gps_mixed_fleet_dataset.csv", index=False)

print("✅ FINAL REALISTIC GPS FLEET CREATED")
print(df["trip_type"].value_counts())
print(df.head())