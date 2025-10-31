'''
This script reads the yield_df.csv file and imports it to MongoDB.
It connects to a local or dockerized MongoDB instance and uploads the data running 
on port 27017.

Requirements:
- pandas
- pymongo

Usage:
- python implement_mongodb.py
'''
import pandas as pd
from pymongo import MongoClient

CSV_FILE='yield_df.csv'
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'farm_yields_database'
COLLECTION_NAME = 'yields'

print("Reading CSV file...")
df = pd.read_csv(CSV_FILE)
print("CSV file read successfully.")
print(f"loaded {len(df)} rows and {len(df.columns)} columns.")

rename_map = {
    "Area": "country",
    "Item": "crop",
    "Year": "year",
    "hg/ha_yield": "yield_hg_per_ha",
    "average_rain_fall_mm_per_year": "average_rainfall_mm_per_year",
    "pesticides_tonnes": "pesticides_tonnes",
    "avg_temp": "avg_temp"
}
df = df.rename(columns=rename_map)

"""Removing unnamed columns if any"""
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

"""convert numeric fields to appropriate types"""
numeric_fields = [
    "year",
    "yield_hg_per_ha",
    "average_rainfall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]
for field in numeric_fields:
    df[field] = pd.to_numeric(df[field], errors='coerce')

print("columns after renaming and type conversion:", list(df.columns))

"""Connecting to MongoDB..."""
print(f"Connecting to MongoDB at {MONGO_URI}...")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

'''clean existing collection'''
print(f"Cleaning existing collection '{COLLECTION_NAME}' if any...")
collection.drop()

"""Inserting data into MongoDB..."""
records = df.to_dict(orient='records')
collection.insert_many(records)
print(f"Inserted {collection.count_documents({})} documents into the collection '{COLLECTION_NAME}' in database '{DB_NAME}'.")

"""Verification Sample"""
print(collection.find_one())

client.close()