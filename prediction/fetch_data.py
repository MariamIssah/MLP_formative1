import requests
from typing import Dict, Optional, List
import json
from datetime import datetime

MYSQL_API_URL = "http://127.0.0.1:8000"    
MONGO_API_URL = "http://127.0.0.1:8001"    
MONGO_FIELD_MAPPING = {
    "year": "year",
    "avg_temp": "avg_temp",
    "average_rain_fall_mm_per_year": "average_rainfall_mm_per_year",
    "pesticides_tonnes": "pesticides_tonnes",
    "hg_ha_yield": "yield_hg_per_ha"
}

def fetch_mysql_latest() -> Optional[Dict]:

    try:
        url = f"{MYSQL_API_URL}/climate-data/latest"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
<<<<<<< HEAD
        print(" Successfully fetched latest record from MySQL API")
        return data
    except requests.RequestException as e:
        print(f" Error fetching from MySQL API: {str(e)}")
=======
        print("Successfully fetched latest record from MySQL API")
        return data
    except requests.RequestException as e:
        print(f"Error fetching from MySQL API: {str(e)}")
>>>>>>> 420d72319ab25d22b424395c46478e00280e2b28
        return None

def fetch_mongodb_latest() -> Optional[Dict]:

    try:
        url = f"{MONGO_API_URL}/yields"  
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
      
        latest = data[0] if data else None
        if latest:
           
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
<<<<<<< HEAD
            print(" Successfully fetched latest record from MongoDB API")
            return mapped_data
        return None
    except requests.RequestException as e:
        print(f" Error fetching from MongoDB API: {str(e)}")
=======
            print("Successfully fetched latest record from MongoDB API")
            return mapped_data
        return None
    except requests.RequestException as e:
        print(f"Error fetching from MongoDB API: {str(e)}")
>>>>>>> 420d72319ab25d22b424395c46478e00280e2b28
        return None

def compare_records(mysql_record: Optional[Dict], mongo_record: Optional[Dict]) -> Dict:
   
    if not mysql_record and not mongo_record:
        return {"status": "error", "message": "Both database fetches failed"}
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "mysql_status": "success" if mysql_record else "failed",
        "mongodb_status": "success" if mongo_record else "failed",
        "discrepancies": []
    }
    
    if mysql_record and mongo_record:
      
        fields_to_compare = [
            "year", "avg_temp", "average_rain_fall_mm_per_year",
            "pesticides_tonnes", "hg_ha_yield", "country_name", "crop_name"
        ]
        
        for field in fields_to_compare:
            mysql_value = mysql_record.get(field)
            mongo_value = mongo_record.get(field)
            if mysql_value != mongo_value:
                comparison["discrepancies"].append({
                    "field": field,
                    "mysql_value": mysql_value,
                    "mongodb_value": mongo_value
                })
    
    return comparison

def save_comparison(comparison: Dict, filename: str = "data_comparison.json"):
  
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
            f.write('\n')
<<<<<<< HEAD
        print(f" Comparison saved to {filename}")
    except Exception as e:
        print(f" Error saving comparison: {str(e)}")

def main():
 
    print("\n Fetching latest records from both APIs...")
   
=======
        print(f"Comparison saved to {filename}")
    except Exception as e:
        print(f"Error saving comparison: {str(e)}")

def main():
    """
    Main function to orchestrate the data fetching and comparison process.
    """
    print("\nFetching latest records from both APIs...")
    
    # Fetch from both APIs
>>>>>>> 420d72319ab25d22b424395c46478e00280e2b28
    mysql_record = fetch_mysql_latest()
    mongo_record = fetch_mongodb_latest()

    comparison = compare_records(mysql_record, mongo_record)
<<<<<<< HEAD
 
    print("\n Comparison Results:")
=======
    
    # Print results
    print("\nComparison Results:")
>>>>>>> 420d72319ab25d22b424395c46478e00280e2b28
    print(f"MySQL Status: {comparison['mysql_status']}")
    print(f"MongoDB Status: {comparison['mongodb_status']}")
    
    if comparison.get("discrepancies"):
<<<<<<< HEAD
        print("\n Discrepancies found:")
=======
        print("\nDiscrepancies found:")
>>>>>>> 420d72319ab25d22b424395c46478e00280e2b28
        for disc in comparison["discrepancies"]:
            print(f"- Field: {disc['field']}")
            print(f"  MySQL: {disc['mysql_value']}")
            print(f"  MongoDB: {disc['mongodb_value']}\n")
    else:
<<<<<<< HEAD
        print("\n No discrepancies found between databases")
=======
        print("\nNo discrepancies found between databases")
>>>>>>> 420d72319ab25d22b424395c46478e00280e2b28
    
    save_comparison(comparison)
   
    return mysql_record if mysql_record else mongo_record

if __name__ == "__main__":
    latest_record = main()
    if latest_record:
<<<<<<< HEAD
        print("\n Latest Record:")
        print(json.dumps(latest_record, indent=2))
    else:
        print("\n Failed to fetch latest record from either database")
=======
        print("\nLatest Record:")
        print(json.dumps(latest_record, indent=2))
    else:

        print("\nFailed to fetch latest record from either database")
>>>>>>> 420d72319ab25d22b424395c46478e00280e2b28
