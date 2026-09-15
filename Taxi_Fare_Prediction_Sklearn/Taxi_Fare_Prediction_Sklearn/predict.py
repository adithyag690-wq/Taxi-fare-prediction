import sys
from pathlib import Path
import joblib
import pandas as pd

# Fix console encoding on Windows for unicode symbols
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "taxi_fare_model.pkl"

if not MODEL_PATH.exists():
    print("Model not found! Please run 'python train_model.py' first to train and save the model.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

# Example trip
trip = pd.DataFrame([{
    "trip_distance_km": 8.5,
    "passenger_count": 2,
    "pickup_hour": 19,
    "is_weekend": 0
}])

fare = model.predict(trip)[0]
try:
    print(f"Predicted taxi fare: ₹{fare:.2f}")
except UnicodeEncodeError:
    print(f"Predicted taxi fare: Rs. {fare:.2f}")

