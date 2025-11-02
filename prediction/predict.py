import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

MYSQL_API_URL = "http://127.0.0.1:8000"    
MONGO_API_URL = "http://127.0.0.1:8001"    
MODEL_PATH = r"C:\Users\awini\MLP_formative1\results\models\crop_prediction.keras"
DATA_CSV = r"C:\Users\awini\MLP_formative1\yield_df.csv"
PREDICTION_OUTPUT = "prediction_output.json"
LOG_PREDICTION = True
FEATURE_ORDER = [
    "year",                      
    "avg_temp",                    
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",           
    "country_name",               
    "crop_name"                    
]

_scaler = None
_label_encoders = None


def initialize_preprocessors():

    global _scaler, _label_encoders
    
    if _scaler is not None and _label_encoders is not None:
        return _scaler, _label_encoders
    
    try:
        print("Loading training data to initialize preprocessors...")
        df = pd.read_csv(DATA_CSV, index_col=0)
        
        df = df.rename(columns={
            "Year": "year",
            "Item": "crop_name",
            "Area": "country_name",
            "hg/ha_yield": "hg_ha_yield"
        })
        
        NUM_COLS = ["year", "avg_temp", "average_rain_fall_mm_per_year", "pesticides_tonnes"]
        CAT_COLS = ["country_name", "crop_name"]
        
        _scaler = StandardScaler()
        X_num = df[NUM_COLS].astype(float).fillna(0.0)
        _scaler.fit(X_num)
        print(f"StandardScaler fitted on {len(NUM_COLS)} numeric features")
        
        _label_encoders = {}
        for col in CAT_COLS:
            le = LabelEncoder()
            values = df[col].astype(str).fillna('')
            le.fit(values)
            _label_encoders[col] = le
            print(f"LabelEncoder fitted for {col} ({len(le.classes_)} unique values)")
        
        return _scaler, _label_encoders
        
    except FileNotFoundError:
        print(f"Warning: Training data not found at {DATA_CSV}")
        print("Falling back to simple preprocessing (may cause inaccurate predictions)")
        return None, None
    except Exception as e:
        print(f"Warning: Error initializing preprocessors: {str(e)}")
        print("Falling back to simple preprocessing (may cause inaccurate predictions)")
        return None, None


def fetch_latest_record():

    try:
        url = f"{MYSQL_API_URL}/climate-data/latest"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError("MySQL API returned empty response")
            
        print("Successfully fetched latest record from MySQL API")
        return data
    except requests.RequestException as e:
        print(f"Warning: Error fetching from MySQL API: {str(e)}")
        print("Attempting fallback to MongoDB API...")
        
        try:
            url = f"{MONGO_API_URL}/yields"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) == 0:
                raise ValueError("MongoDB API returned empty response")
            
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
            print("Successfully fetched latest record from MongoDB API")
            return mapped_data
            
        except requests.RequestException as e:
            print(f"Error fetching from MongoDB API: {str(e)}")
            raise Exception("Failed to fetch data from both MySQL and MongoDB APIs")
        except ValueError as e:
            print(f" {str(e)}")
            raise

def validate_record(record):

    required_fields = FEATURE_ORDER + ["record_id"]
    missing_fields = []
    
    for field in required_fields:
        if field not in record or record[field] is None:
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def preprocess_data(record, scaler=None, label_encoders=None):
    
    is_valid, missing_fields = validate_record(record)
    if not is_valid:
        print(f"Warning: Missing required fields: {missing_fields}")
        print("Attempting to handle missing data with defaults...")
    
    warnings = []
    
    NUM_COLS = ["year", "avg_temp", "average_rain_fall_mm_per_year", "pesticides_tonnes"]
    CAT_COLS = ["country_name", "crop_name"]
    
    num_values = []
    for col in NUM_COLS:
        val = record.get(col)
        
        if val is None:
            warnings.append(f"Missing numerical field {col}, using default: 0.0")
            val = 0.0
        elif isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                warnings.append(f"Invalid value for {col}: '{val}', using default: 0.0")
                val = 0.0
        else:
            try:
                val = float(val)
            except (TypeError, ValueError):
                warnings.append(f"Cannot convert {col} to float: {val}, using default: 0.0")
                val = 0.0
        
        if col == "year" and (val < 1900 or val > 2100):
            warnings.append(f"Year {val} seems out of range")
        elif col == "avg_temp" and (val < -50 or val > 60):
            warnings.append(f"Average temperature {val} seems out of range")
        elif col == "average_rain_fall_mm_per_year" and (val < 0 or val > 10000):
            warnings.append(f"Rainfall {val} seems out of range")
        elif col == "pesticides_tonnes" and (val < 0 or val > 100000):
            warnings.append(f"Pesticides {val} seems out of range")
        
        num_values.append(val)
    
    cat_values = []
    for col in CAT_COLS:
        val = record.get(col)
        
        if val is None or val == "":
            if col == "country_name":
                val = "Ghana"  
            elif col == "crop_name":
                val = "Maize"  
            warnings.append(f"Missing {col}, using default: {val}")
        
       
        if label_encoders and col in label_encoders:
            try:
               
                encoded_val = label_encoders[col].transform([str(val)])[0]
                cat_values.append(float(encoded_val))
            except ValueError:
                
                warnings.append(f"Unknown {col} '{val}', using first available class")
                encoded_val = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
                cat_values.append(float(encoded_val))
        else:
           
            if col == "country_name":
                country_mapping = {"Ghana": 1.0, "Kenya": 2.0, "Nigeria": 3.0}
                cat_values.append(country_mapping.get(str(val), 1.0))
            elif col == "crop_name":
                crop_mapping = {"Maize": 1.0, "Rice": 2.0, "Wheat": 3.0}
                cat_values.append(crop_mapping.get(str(val), 1.0))
    
    if scaler is not None:
        try:
            num_array = np.array(num_values, dtype=float).reshape(1, -1)
            num_scaled = scaler.transform(num_array)
            num_values = num_scaled[0].tolist()
        except Exception as e:
            warnings.append(f"Error scaling numeric features: {str(e)}, using unscaled values")
    
    all_values = num_values + cat_values
    
    if warnings:
        print("Data Quality Warnings:")
        for warning in warnings:
            print(f"    - {warning}")
   
    try:
        X = np.array(all_values, dtype=float).reshape(1, -1)
        
        expected_features = len(NUM_COLS) + len(CAT_COLS)
        if X.shape[1] != expected_features:
            raise ValueError(f"Feature vector shape {X.shape} does not match expected features {expected_features}")
        
        metadata = {
            "country_name": record.get("country_name", "Unknown"),
            "crop_name": record.get("crop_name", "Unknown"),
            "year": record.get("year", None),
            "avg_temp": record.get("avg_temp", None),
            "average_rain_fall_mm_per_year": record.get("average_rain_fall_mm_per_year", None),
            "pesticides_tonnes": record.get("pesticides_tonnes", None),
            "record_id": record.get("record_id", None)
        }
        
        return X, metadata
    except Exception as e:
        raise ValueError(f"Error creating feature vector: {str(e)}")

def load_keras_model():
 
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            f"Please ensure the model file 'crop_prediction.keras' exists in the results/models/ directory."
        )
    
    print(f"Model path: {MODEL_PATH}")
    
    if not MODEL_PATH.endswith('.keras') and not MODEL_PATH.endswith('.h5'):
        print(f"Warning: Model file doesn't have standard extension (.keras or .h5)")
    
    try:
      
        model = load_model(MODEL_PATH)
        print(f" Model loaded successfully (crop_prediction.keras)")
    
        input_shape = model.input_shape if hasattr(model, 'input_shape') else "Unknown"
        print(f"Model input shape: {input_shape}")
        
        return model
    except OSError as e:
        raise OSError(f"Error loading model file: {str(e)}\nModel may be corrupted or in wrong format.")
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

def log_prediction_to_api(record_id, prediction):

    url = f"{MYSQL_API_URL}/predictions"
    
    payload = {
        "record_id": record_id,      
        "prediction": float(prediction)  
    }
    
    try:
    
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()  
        
        print("Prediction logged to MySQL API successfully.")
        return res.json()
    
    except Exception as e:
        print(f"Could not log prediction to MySQL API: {e}")
        return None

def save_local_log(metadata, prediction):

    try:

        data = {
            "timestamp": datetime.now().isoformat(), 
            "country": metadata.get("country_name", "Unknown"),
            "crop": metadata.get("crop_name", "Unknown"),
            "year": metadata.get("year"),
            "avg_temp": metadata.get("avg_temp"),
            "average_rainfall_mm_per_year": metadata.get("average_rain_fall_mm_per_year"),
            "pesticides_tonnes": metadata.get("pesticides_tonnes"),
            "record_id": metadata.get("record_id"),
            "predicted_yield_hg_per_ha": float(prediction),
            "original_record": metadata  
        }
        
        with open(PREDICTION_OUTPUT, "a", encoding="utf-8") as f:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str + "\n")
        
        print(f"Prediction saved to {PREDICTION_OUTPUT}")
    except IOError as e:
        raise IOError(f"Error writing to file {PREDICTION_OUTPUT}: {str(e)}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Error serializing data to JSON: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error saving local log: {str(e)}")

def main():
  
    try:
        print("=" * 60)
        print("TASK 3: Crop Yield Prediction Pipeline")
        print("=" * 60)
        
        print("\n Step 1: Fetching latest climate data record from API...")
        try:
            record = fetch_latest_record()
            if not record:
                raise ValueError("No data record received from API")
            print("Successfully fetched record")
            print("Latest Record:")
            print(json.dumps(record, indent=2))
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise

        print("\n Step 2: Initializing preprocessors...")
        scaler, label_encoders = initialize_preprocessors()
        if scaler is None or label_encoders is None:
            print("Warning: Using fallback preprocessing (predictions may be inaccurate)")
        else:
            print("Preprocessors initialized successfully")
        
        print("\n Step 3: Preparing data for model prediction...")
        try:
            X, metadata = preprocess_data(record, scaler=scaler, label_encoders=label_encoders)
            print("Data preprocessing completed")
            print(f"Feature Vector Shape: {X.shape}")
            print(f"Feature Vector: {X.tolist()}")
            print(f"\nPrediction Details:")
            print(f"   Country: {metadata.get('country_name', 'Unknown')}")
            print(f"   Crop: {metadata.get('crop_name', 'Unknown')}")
            print(f"   Year: {metadata.get('year', 'Unknown')}")
            print(f"   Average Temperature: {metadata.get('avg_temp', 'Unknown')}°C")
            print(f"   Average Rainfall: {metadata.get('average_rain_fall_mm_per_year', 'Unknown')} mm/year")
            print(f"   Pesticides: {metadata.get('pesticides_tonnes', 'Unknown')} tonnes")
        except ValueError as e:
            print(f"Error preprocessing data: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error during preprocessing: {str(e)}")
            raise

        print("\nStep 4: Loading Keras model...")
        try:
            model = load_keras_model()
            print("Model loaded successfully")
        except FileNotFoundError as e:
            print(f"Model file not found: {str(e)}")
            raise
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        print("\nStep 5: Making prediction...")
        try:
            prediction = model.predict(X, verbose=0)
            
            if prediction.ndim == 2:
                prediction_value = float(prediction[0][0])
            else:
                prediction_value = float(prediction[0])
            
            raw_prediction = prediction_value
            prediction_value = max(0.0, prediction_value) 
            
            print(f"Prediction completed")
            print(f"\nPrediction Results:")
            print(f"   Raw prediction: {raw_prediction:.4f} hg/ha")
            if raw_prediction < 0:
                print(f"Negative prediction clamped to 0")
            print(f"Final predicted yield: {prediction_value:.4f} hg/ha")
            print(f"\nYield Prediction Summary:")
            print(f"Country: {metadata.get('country_name', 'Unknown')}")
            print(f"Crop Type: {metadata.get('crop_name', 'Unknown')}")
            print(f"Predicted Yield: {prediction_value:.2f} hg/ha ({prediction_value/100:.2f} tonnes/ha)")
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise

        print("\nStep 6: Logging results...")
        record_id = metadata.get("record_id", None)
        
        if LOG_PREDICTION and record_id:
            try:
                api_result = log_prediction_to_api(record_id, prediction_value)
                if api_result:
                    print("Results logged to database successfully")
            except Exception as e:
                print(f"Warning: Could not log to API: {str(e)}")
                print("Continuing with local log only...")
        elif not record_id:
            print("Warning: No record_id found, skipping API logging")
            
        try:
            save_local_log(metadata, prediction_value)
            print("Results saved to local file")
        except Exception as e:
            print(f"Error saving local log: {str(e)}")
            raise

        print("\n" + "=" * 60)
        print("TASK 3 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n  Process interrupted by user")
        raise
    except Exception as e:
        print("\n" + "=" * 60)
        print(f" TASK 3 FAILED: {str(e)}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    """
    Standard Python idiom to ensure main() only runs when script is executed directly,
    not when imported as a module.
    """
    main()
