import pandas as pd
import mysql.connector

# 1️ Connect to your MySQL Database 
conn = mysql.connector.connect(
    host="localhost",
    user="root",                      # your MySQL username
    password="G16Formative1",         # your MySQL password
    database="agri_yield_db"          # your database name
)
cursor = conn.cursor()

# 2️ Load CSV Dataset 
df = pd.read_csv("yield_df.csv")

# Display first few rows (optional)
print(df.head())

# 3️ Map dataset columns to database fields
# (Make sure these names match the actual columns in yield_df.csv)
for _, row in df.iterrows():
    country = row["Area"]                                   # corresponds to "country"
    crop = row["Item"]                                      # corresponds to "crop"
    year = int(row["Year"])
    avg_temp = float(row["avg_temp"])
    avg_rain = float(row["average_rain_fall_mm_per_year"])
    pesticides = float(row["pesticides_tonnes"])
    yield_value = float(row["hg/ha_yield"])                 # note the slash here

    # Insert or get country_id
    cursor.execute("SELECT country_id FROM countries WHERE country_name=%s", (country,))
    result = cursor.fetchone()
    if result:
        country_id = result[0]
    else:
        cursor.execute("INSERT INTO countries (country_name) VALUES (%s)", (country,))
        conn.commit()
        country_id = cursor.lastrowid

    # Insert or get crop_id
    cursor.execute("SELECT crop_id FROM crops WHERE crop_name=%s", (crop,))
    result = cursor.fetchone()
    if result:
        crop_id = result[0]
    else:
        cursor.execute("INSERT INTO crops (crop_name) VALUES (%s)", (crop,))
        conn.commit()
        crop_id = cursor.lastrowid

    # 4️ Insert data into climate_data 
    insert_query = """
        INSERT INTO climate_data
        (country_id, crop_id, year, avg_temp, average_rain_fall_mm_per_year, pesticides_tonnes, hg_ha_yield)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (country_id, crop_id, year, avg_temp, avg_rain, pesticides, yield_value))
    conn.commit()

print("✅ Data imported successfully into MySQL!")

# Close connection
cursor.close()
conn.close()
