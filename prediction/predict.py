import os
import json
import requests
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# Configuration

API_URL = "http://127.0.0.1:8000"
MODEL_PATH = r"C:\Users\awini\MLP_formative1\results\models\crop_prediction.keras"
PREDICTION_OUTPUT = "prediction_output.json"
LOG_PREDICTION = True  

# Fields from the API — ensure order matches what your model expects
FEATURE_ORDER = [
    "year",
    "avg_temp",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes"
]

# Helper Functions

def fetch_latest_record():
    """Fetch latest climate record from FastAPI."""
    url = f"{API_URL}/climate-data/latest"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def preprocess_data(record):
    """Convert JSON record into NumPy array suitable for model prediction."""
    values = []
    for feature in FEATURE_ORDER:
        val = record.get(feature, 0.0)
        if val is None:
            val = 0.0
        values.append(float(val))
    X = np.array(values, dtype=float).reshape(1, -1)
    return X

def load_keras_model():
    """Load the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    print(f" Loaded model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    return model

def log_prediction_to_api(record_id, prediction):
    """Send prediction result back to FastAPI (optional)."""
    url = f"{API_URL}/predictions"
    payload = {"record_id": record_id, "prediction": float(prediction)}
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print(" Prediction logged to API successfully.")
        return res.json()
    except Exception as e:
        print(" Could not log prediction to API:", e)
        return None

def save_local_log(record, prediction):
    """Save the prediction result locally in JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "record": record,
        "prediction": float(prediction)
    }
    with open(PREDICTION_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
    print(f" Prediction saved to {PREDICTION_OUTPUT}")

def main():
    print(" Fetching latest climate data record from API...")
    record = fetch_latest_record()
    print(" Latest Record:", json.dumps(record, indent=2))

    print("\n Preparing data for model prediction...")
    X = preprocess_data(record)
    print(" Feature Vector:", X.tolist())

    print("\n Loading Keras model...")
    model = load_keras_model()

    print("\n Making prediction...")
    prediction = model.predict(X)
    prediction_value = float(prediction[0][0]) if prediction.ndim == 2 else float(prediction[0])
    print(f" Predicted yield (hg/ha): {prediction_value}")

    record_id = record.get("record_id", None)
    if LOG_PREDICTION and record_id:
        log_prediction_to_api(record_id, prediction_value)

    save_local_log(record, prediction_value)
    print("\n Task 3 completed successfully!")

if _name_ == "_main_":
    main()