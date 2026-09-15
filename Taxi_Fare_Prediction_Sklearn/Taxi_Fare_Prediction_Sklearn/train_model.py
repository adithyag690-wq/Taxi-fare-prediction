import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "taxi_fares.csv"
MODEL_PATH = BASE_DIR / "models" / "taxi_fare_model.pkl"
OUTPUT_PATH = BASE_DIR / "outputs" / "feature_importance.png"

df = pd.read_csv(DATA_PATH)

features = [
    "trip_distance_km",
    "passenger_count",
    "pickup_hour",
    "is_weekend"
]
target = "fare_amount"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=12
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("Taxi Fare Prediction Results")
print("-" * 35)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

os.makedirs(MODEL_PATH.parent, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# Feature importance chart
importance = pd.Series(model.feature_importances_, index=features).sort_values()
importance.plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
plt.savefig(OUTPUT_PATH)
plt.close()
print(f"Chart saved to: {OUTPUT_PATH}")
