import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

MODEL_PATH = r"C:\Users\awini\MLP_formative1\results\models\crop_prediction.keras"
DATA_CSV = r"C:\Users\awini\MLP_formative1\yield_df.csv"
FEATURE_ORDER = [
    "year",
    "avg_temp",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "country_name",
    "crop_name"
]

NUM_COLS = ["year", "avg_temp", "average_rain_fall_mm_per_year", "pesticides_tonnes"]
CAT_COLS = ["country_name", "crop_name"]

print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("\nModel summary:\n")
model.summary()

print("\nLoading dataset for scaler/encoder estimation:", DATA_CSV)
df = pd.read_csv(DATA_CSV, index_col=0)

cols = df.columns.tolist()
print("Dataset columns sample:", cols[:10])

assert all(col in df.columns for col in ["Year", "avg_temp", "average_rain_fall_mm_per_year", "pesticides_tonnes", "Item"]), "CSV missing expected columns"

df = df.rename(columns={
    "Year": "year",
    "Item": "crop_name",
    "Area": "country_name",
    "hg/ha_yield": "hg_ha_yield"
})

print("\nNumeric feature stats (mean, std):")
scaler = StandardScaler()
X_num = df[NUM_COLS].astype(float).fillna(0.0)
scaler.fit(X_num)
for c, m, s in zip(NUM_COLS, scaler.mean_, np.sqrt(scaler.var_)):
    print(f" - {c}: mean={m:.4f}, std={s:.4f}")

label_encoders = {}
for c in CAT_COLS:
    le = LabelEncoder()
    values = df[c].astype(str).fillna('')
    le.fit(values)
    label_encoders[c] = le
    print(f"\nLabel encoder for {c}: classes sample={list(le.classes_)[:10]}")

record = {
    "record_id": 28247,
    "year": 2023,
    "avg_temp": 26.5,
    "average_rain_fall_mm_per_year": 820.0,
    "pesticides_tonnes": 18.3,
    "hg_ha_yield": 4200.6,
    "country_name": "Ghana",
    "crop_name": "Maize"
}

num_vals = np.array([record[c] for c in NUM_COLS], dtype=float).reshape(1, -1)
num_scaled = scaler.transform(num_vals)
cat_vals = np.array([label_encoders[c].transform([record[c]])[0] for c in CAT_COLS], dtype=float).reshape(1, -1)
X = np.hstack([num_scaled, cat_vals])
print("\nPrepared feature vector (scaled numeric + encoded cats):")
print(X)

pred = model.predict(X)
print("\nRaw model output:", pred)

prediction_value = float(pred[0][0]) if pred.ndim == 2 else float(pred[0])
print("Prediction value:", prediction_value)

country_mapping = {"Ghana":1.0, "Kenya":2.0, "Nigeria":3.0}
crop_mapping = {"Maize":1.0, "Rice":2.0, "Wheat":3.0}
vals_simple = np.array([
    record['year'], record['avg_temp'], record['average_rain_fall_mm_per_year'], record['pesticides_tonnes'],
    country_mapping.get(record['country_name'], 0.0), crop_mapping.get(record['crop_name'], 0.0)
], dtype=float).reshape(1, -1)
print("\nSimple feature vector (no scaling, simple mapping):")
print(vals_simple)
pred_simple = model.predict(vals_simple)
print("Raw model output for simple vector:", pred_simple)

print("\nDone.")
