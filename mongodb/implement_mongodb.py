'''
This script reads the yield_df.csv file and imports it to MongoDB.
It connects to a local or dockerized MongoDB instance and uploads the data running 
on port 27017.

Requirements:
- pandas
- pymongo

Usage:
- python implement_mongodb.py

Features:
- Data validation and type conversion
- Comprehensive error handling
- Progress logging
- Data verification
'''

import pandas as pd
from pymongo import MongoClient
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mongodb_import.log')
    ]
)
logger = logging.getLogger(__name__)

CSV_FILE = 'yield_df.csv'
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'farm_yields_database'
COLLECTION_NAME = 'yields'

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate the DataFrame structure and required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if validation passes
    """
    required_columns = ['Area', 'Item', 'Year', 'hg/ha_yield', 
                       'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    if df.empty:
        logger.error("DataFrame is empty")
        return False
        
    logger.info("DataFrame validation successful")
    return True

def main():
    try:
        # Read and validate CSV file
        logger.info("Reading CSV file...")
        df = pd.read_csv(CSV_FILE)
        logger.info(f"CSV file read successfully. Loaded {len(df)} rows and {len(df.columns)} columns.")
        
        if not validate_dataframe(df):
            sys.exit(1)
            
        # Column renaming for consistency
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
        
        # Data cleaning
        logger.info("Cleaning data...")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Type conversion with detailed logging
        numeric_fields = [
            "year", 
            "yield_hg_per_ha",
            "average_rainfall_mm_per_year", 
            "pesticides_tonnes",
            "avg_temp"
        ]
        
        for field in numeric_fields:
            original_count = len(df[field])
            df[field] = pd.to_numeric(df[field], errors='coerce')
            null_count = df[field].isnull().sum()
            if null_count > 0:
                logger.warning(f"Field '{field}': {null_count} non-numeric values converted to NaN")
        
        logger.info(f"Columns after processing: {list(df.columns)}")
        
        # MongoDB operations
        logger.info(f"Connecting to MongoDB at {MONGO_URI}...")
        client = MongoClient(MONGO_URI)
        
        # Verify connection
        client.admin.command('ismaster')
        logger.info("MongoDB connection established successfully")
        
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Clean existing collection
        logger.info(f"Cleaning existing collection '{COLLECTION_NAME}'...")
        collection.drop()
        logger.info("Collection cleaned successfully")
        
        # Insert data
        logger.info("Converting DataFrame to records...")
        records = df.to_dict(orient='records')
        
        logger.info(f"Inserting {len(records)} documents into MongoDB...")
        result = collection.insert_many(records)
        logger.info(f"Successfully inserted {len(result.inserted_ids)} documents")
        
        # Verification and statistics
        total_documents = collection.count_documents({})
        logger.info(f"Total documents in collection: {total_documents}")
        
        # Sample document for verification
        sample = collection.find_one()
        logger.info("Sample document verified successfully")
        logger.debug(f"Sample document: {sample}")
        
        # Collection statistics
        stats = db.command('collstats', COLLECTION_NAME)
        logger.info(f"Collection size: {stats['size']} bytes")
        logger.info(f"Document count: {stats['count']}")
        
    except FileNotFoundError:
        logger.error(f"CSV file '{CSV_FILE}' not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")

if __name__ == "__main__":
    main()
