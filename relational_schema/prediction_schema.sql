-- Predictions table
USE agri_yield_db;

CREATE TABLE predictions
(
    prediction_id INT
    AUTO_INCREMENT PRIMARY KEY,
    record_id INT NOT NULL,
    prediction_value FLOAT NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY
    (record_id) REFERENCES climate_data
    (record_id)
);

    -- Create index for faster lookups
    CREATE INDEX idx_record_id ON predictions(record_id);

    -- Stored procedure for adding predictions
    DELIMITER //
    CREATE PROCEDURE AddPrediction(
    IN p_record_id INT,
    IN p_prediction FLOAT
)
    BEGIN
        INSERT INTO predictions
            (record_id, prediction_value)
        VALUES
            (p_record_id, p_prediction);
    END
    //
DELIMITER ;