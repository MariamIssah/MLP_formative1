"""
Crop Yield Prediction Script

This script demonstrates a complete ML pipeline for predicting crop yields:
1. Fetches latest climate data from a REST API
2. Preprocesses the data for model consumption
3. Loads a trained Keras model
4. Makes predictions
5. Logs results both locally and to an API

Purpose: To demonstrate ML model deployment and inference patterns
"""

import os
import json
import requests
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# CONFIGURATION SECTION

# API endpoints for climate data
MYSQL_API_URL = "http://127.0.0.1:8000"    # MySQL FastAPI server
MONGO_API_URL = "http://127.0.0.1:8001"    # MongoDB FastAPI server

# Path to the pre-trained Keras model - using raw string for Windows path
MODEL_PATH = r"C:\Users\awini\MLP_formative1\results\models\crop_prediction.keras"

# Local file for storing prediction results
PREDICTION_OUTPUT = "prediction_output.json"

# Flag to control whether predictions are sent back to API
LOG_PREDICTION = True

# Feature ordering must match the model's expected input structure
# This ensures consistent data preprocessing between training and inference
FEATURE_ORDER = [
    "year",                        # Temporal feature - year of observation
    "avg_temp",                    # Climate feature - average temperature
    "average_rain_fall_mm_per_year", # Climate feature - annual rainfall
    "pesticides_tonnes",           # Agricultural input - pesticide usage
    "country_name",                # Geographical feature - country
    "crop_name"                    # Agricultural feature - crop type
]


# HELPER FUNCTIONS


def fetch_latest_record():
    """
    Fetches the most recent climate record from both MySQL and MongoDB APIs.
    Prefers MySQL data but falls back to MongoDB if MySQL fails.
    
    Returns:
        dict: JSON response containing climate data record
        
    Raises:
        requests.exceptions.RequestException: If both API requests fail
        ValueError: If responses cannot be parsed as JSON
    """
    # First try MySQL API
    try:
        url = f"{MYSQL_API_URL}/climate-data/latest"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print(" ✅ Successfully fetched latest record from MySQL API")
        return response.json()
    except requests.RequestException as e:
        print(f" ❌ Error fetching from MySQL API: {str(e)}")
        
        # Fall back to MongoDB API
        try:
            url = f"{MONGO_API_URL}/yields"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get the most recent record and map fields
            if data:
                latest = data[0]
                mapped_data = {
                    "record_id": latest.get("_id"),
                    "year": latest.get("year"),
                    "avg_temp": latest.get("avg_temp"),
                    "average_rain_fall_mm_per_year": latest.get("average_rainfall_mm_per_year"),
                    "pesticides_tonnes": latest.get("pesticides_tonnes"),
                    "hg_ha_yield": latest.get("yield_hg_per_ha"),
                    "country_name": latest.get("country"),
                    "crop_name": latest.get("crop")
                }
                print(" ✅ Successfully fetched latest record from MongoDB API")
                return mapped_data
            
        except requests.RequestException as e:
            print(f" ❌ Error fetching from MongoDB API: {str(e)}")
            raise Exception("Failed to fetch data from both MySQL and MongoDB APIs")

def preprocess_data(record):
    """
    Converts JSON record into properly formatted NumPy array for model prediction.
    Handles both numerical and categorical features.
    
    Args:
        record (dict): Climate data record from API
        
    Returns:
        numpy.ndarray: Feature vector shaped (1, n_features) for model input
    """
    values = []
    
    # Country encoding (one-hot or label encoding would be better in production)
    country_mapping = {
        "Ghana": 1.0,
        "Kenya": 2.0,
        "Nigeria": 3.0
        # Add other countries as needed
    }
    
    # Crop encoding
    crop_mapping = {
        "Maize": 1.0,
        "Rice": 2.0,
        "Wheat": 3.0
        # Add other crops as needed
    }
    
    # Iterate through features in predefined order to maintain consistency
    for feature in FEATURE_ORDER:
        # Handle different feature types
        if feature == "country_name":
            country = record.get(feature, "Ghana")  # Default to Ghana if missing
            val = country_mapping.get(country, 0.0)  # Default to 0.0 if country not in mapping
        elif feature == "crop_name":
            crop = record.get(feature, "Maize")  # Default to Maize if missing
            val = crop_mapping.get(crop, 0.0)  # Default to 0.0 if crop not in mapping
        else:
            # Handle numerical features as before
            val = record.get(feature, 0.0)
            if val is None:
                val = 0.0
            val = float(val)
        
        values.append(val)
    
    # Convert list to NumPy array and reshape for single prediction
    X = np.array(values, dtype=float).reshape(1, -1)
    return X

def load_keras_model():
    """
    Loads the pre-trained Keras model from disk.
    
    Returns:
        tf.keras.Model: Loaded Keras model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        OSError: If model file cannot be loaded
    """
    # Check if model file exists before attempting to load
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    
    print(f" Loaded model from {MODEL_PATH}")
    
    # Load and return the Keras model
    model = load_model(MODEL_PATH)
    return model

def log_prediction_to_api(record_id, prediction):
    """
    Sends prediction result back to MySQL API for storage and monitoring.
    
    Args:
        record_id: Unique identifier for the climate record
        prediction (float): Predicted crop yield value
        
    Returns:
        dict: API response if successful, None otherwise
    """
    url = f"{MYSQL_API_URL}/predictions"
    
    # Prepare payload for API request
    payload = {
        "record_id": record_id,      # Link prediction to original record
        "prediction": float(prediction)  # Ensure float type for JSON serialization
    }
    
    try:
        # Send POST request to prediction logging endpoint
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()  # Check for HTTP errors
        
        print(" ✅ Prediction logged to MySQL API successfully.")
        return res.json()
    
    except Exception as e:
        # Gracefully handle API communication failures
        print(f" ❌ Could not log prediction to MySQL API: {e}")
        return None

def save_local_log(record, prediction):
    """
    Appends prediction results to local JSON file for backup and analysis.
    
    Args:
        record (dict): Original climate data record
        prediction (float): Model prediction result
    """
    # Structure data for logging with timestamp
    data = {
        "timestamp": datetime.now().isoformat(),  # ISO format for standardization
        "record": record,                         # Original input data
        "prediction": float(prediction)           # Model output
    }
    
    # Append to file in JSON Lines format (one JSON object per line)
    with open(PREDICTION_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
    
    print(f" Prediction saved to {PREDICTION_OUTPUT}")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Main execution function that orchestrates the prediction pipeline.
    """
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
    
    # Handle different prediction output formats (2D array vs 1D array)
    prediction_value = float(prediction[0][0]) if prediction.ndim == 2 else float(prediction[0])
    
    # Post-process: Ensure non-negative yield predictions
    prediction_value = max(0.0, prediction_value)  # Crop yields cannot be negative
    print(f" Predicted yield (hg/ha): {prediction_value}")

    # Extract record ID for API logging
    record_id = record.get("record_id", None)
    
    # Conditionally log to API if enabled and record_id exists
    if LOG_PREDICTION and record_id:
        log_prediction_to_api(record_id, prediction_value)

    # Always save locally for backup
    save_local_log(record, prediction_value)
    print("\n Task 3 completed successfully!")

if __name__ == "__main__":
    """
    Standard Python idiom to ensure main() only runs when script is executed directly,
    not when imported as a module.
    """
    main()
