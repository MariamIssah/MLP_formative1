from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
from mysql.connector import Error
from typing import List, Optional
import os
from dotenv import load_dotenv
from models import ClimateData


# Load environment variables
load_dotenv()

app = FastAPI(title="Agricultural Yield API")

# Database connection configuration
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.getenv("MYSQL_PASSWORD"),  # Store password in .env
            database="agri_yield_db"
        )
        return connection
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Pydantic model using names instead of IDs
class ClimateData(BaseModel):
    country_name: str
    crop_name: str
    year: int
    avg_temp: float
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    hg_ha_yield: float


# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to Agricultural Yield API"}


# Helper to get country_id and crop_id from names
def get_country_and_crop_ids(cursor, country_name: str, crop_name: str):
    cursor.execute("SELECT country_id FROM countries WHERE country_name = %s", (country_name,))
    country = cursor.fetchone()
    if not country:
        raise HTTPException(status_code=404, detail=f"Country '{country_name}' not found")

    cursor.execute("SELECT crop_id FROM crops WHERE crop_name = %s", (crop_name,))
    crop = cursor.fetchone()
    if not crop:
        raise HTTPException(status_code=404, detail=f"Crop '{crop_name}' not found")

    # Return as tuple of integers
    return country["country_id"], crop["crop_id"]


# READ - Get all countries
@app.get("/countries", response_model=List[dict])
async def get_countries():
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM countries")
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


# READ - Get all crops
@app.get("/crops", response_model=List[dict])
async def get_crops():
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM crops")
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


# CREATE - Add new climate data (using names, not IDs)
@app.post("/climate-data", response_model=dict)
async def add_climate_data(data: ClimateData):
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)

        # Get IDs from names
        country_id, crop_id = get_country_and_crop_ids(cursor, data.country_name, data.crop_name)

        # Insert new record using stored procedure (if exists)
        try:
            cursor.callproc('AddYieldRecord', [
                country_id,
                crop_id,
                data.year,
                data.avg_temp,
                data.average_rain_fall_mm_per_year,
                data.pesticides_tonnes,
                data.hg_ha_yield
            ])
        except Error:
            # If stored procedure doesn’t exist, insert manually
            insert_query = """
                INSERT INTO climate_data (country_id, crop_id, year, avg_temp, average_rain_fall_mm_per_year, pesticides_tonnes, hg_ha_yield)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                country_id,
                crop_id,
                data.year,
                data.avg_temp,
                data.average_rain_fall_mm_per_year,
                data.pesticides_tonnes,
                data.hg_ha_yield
            ))

        conn.commit()
        return {"message": " Climate data added successfully"}
    except Error as e:
        conn.rollback()
        print(" MYSQL ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"MySQL Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()


# READ - Fetch climate data with filters
@app.get("/climate-data", response_model=List[dict])
async def get_climate_data(
    country_name: Optional[str] = None,
    crop_name: Optional[str] = None,
    year: Optional[int] = None
):
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT cd.*, c.country_name, cr.crop_name
            FROM climate_data cd
            JOIN countries c ON cd.country_id = c.country_id
            JOIN crops cr ON cd.crop_id = cr.crop_id
            WHERE 1=1
        """
        params = []

        if country_name:
            query += " AND c.country_name = %s"
            params.append(country_name)
        if crop_name:
            query += " AND cr.crop_name = %s"
            params.append(crop_name)
        if year:
            query += " AND cd.year = %s"
            params.append(year)

        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


# UPDATE - Update existing record by ID (using names)
@app.put("/climate-data/{record_id}")
async def update_climate_data(record_id: int, data: ClimateData):
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)

        # Check if the record exists before updating
        cursor.execute("SELECT * FROM climate_data WHERE record_id = %s", (record_id,))
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail=f"Record with ID {record_id} not found")

        # Get country and crop IDs
        country_id, crop_id = get_country_and_crop_ids(cursor, data.country_name, data.crop_name)

        # Perform the update
        update_query = """
            UPDATE climate_data
            SET country_id = %s,
                crop_id = %s,
                year = %s,
                avg_temp = %s,
                average_rain_fall_mm_per_year = %s,
                pesticides_tonnes = %s,
                hg_ha_yield = %s
            WHERE record_id = %s
        """
        cursor.execute(update_query, (
            country_id,
            crop_id,
            data.year,
            data.avg_temp,
            data.average_rain_fall_mm_per_year,
            data.pesticides_tonnes,
            data.hg_ha_yield,
            record_id
        ))

        conn.commit()

        # Confirm update success
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="No rows were updated. Record may not exist.")

        return {"message": f" Climate data with ID {record_id} updated successfully"}

    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"MySQL Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()


#  DELETE - Remove a climate data record
@app.delete("/climate-data/{record_id}", response_model=dict)
async def delete_climate_data(record_id: int):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM climate_data WHERE record_id = %s", (record_id,))
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        return {"message": "Climate data deleted successfully"}
    except Error as e:
        conn.rollback()
        print("MYSQL ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"MySQL Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()