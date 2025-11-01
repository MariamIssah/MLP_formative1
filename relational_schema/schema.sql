-- Create the database
CREATE DATABASE agri_yield_db;
USE agri_yield_db;

-- 1 Table: Countries
CREATE TABLE countries
(
    country_id INT
    AUTO_INCREMENT PRIMARY KEY,
    country_name VARCHAR
    (100) UNIQUE NOT NULL
);

    -- 2 Table: Crops
    CREATE TABLE crops
    (
        crop_id INT
        AUTO_INCREMENT PRIMARY KEY,
    crop_name VARCHAR
        (100) UNIQUE NOT NULL
);

        -- 3️ Table: Climate and Yield Data
        CREATE TABLE climate_data
        (
            record_id INT
            AUTO_INCREMENT PRIMARY KEY,
    country_id INT,
    crop_id INT,
    year YEAR NOT NULL,
    avg_temp FLOAT,
    average_rain_fall_mm_per_year FLOAT,
    pesticides_tonnes FLOAT,
    hg_ha_yield FLOAT,
    FOREIGN KEY
            (country_id) REFERENCES countries
            (country_id),
    FOREIGN KEY
            (crop_id) REFERENCES crops
            (crop_id)
);

            -- 4️ Table: Logs for Triggers
            CREATE TABLE log_records
            (
                log_id INT
                AUTO_INCREMENT PRIMARY KEY,
    record_id INT,
    action_type VARCHAR
                (50),
    action_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

--  STORED PROCEDURE: Add a yield record safely
DELIMITER //
                CREATE PROCEDURE AddYieldRecord(
    IN p_country_id INT,
    IN p_crop_id INT,
    IN p_year YEAR,
    IN p_temp FLOAT,
    IN p_rain FLOAT,
    IN p_pesticides FLOAT,
    IN p_yield FLOAT
)
                BEGIN
                    IF p_year < 1900 OR p_year > YEAR(CURDATE()) THEN
        SIGNAL SQLSTATE '45000'
                    SET MESSAGE_TEXT
                    = 'Invalid year';
                ELSE
                INSERT INTO climate_data
                    (country_id, crop_id, year, avg_temp, average_rain_fall_mm_per_year, pesticides_tonnes, hg_ha_yield)
                VALUES
                    (p_country_id, p_crop_id, p_year, p_temp, p_rain, p_pesticides, p_yield);
                END
                IF;
END //
DELIMITER ;

--  TRIGGER: Log when new data is inserted
DELIMITER //
                CREATE TRIGGER after_insert_climate
AFTER
                INSERT ON
                climate_data
                FOR
                EACH
                ROW
                BEGIN
                    INSERT INTO log_records
                        (record_id, action_type)
                    VALUES
                        (NEW.record_id, 'INSERT');
                END
                //
DELIMITER ;

                --  Sample data
                INSERT INTO countries
                    (country_name)
                VALUES
                    ('Albania'),
                    ('Ghana'),
                    ('Kenya');
                INSERT INTO crops
                    (crop_name)
                VALUES
                    ('Maize'),
                    ('Rice, paddy'),
                    ('Potatoes');
