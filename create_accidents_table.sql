-- Create US Accidents table
CREATE TABLE IF NOT EXISTS us_accidents (
    id VARCHAR(10) PRIMARY KEY,
    source VARCHAR(10),
    severity INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    start_lat DECIMAL(10, 8),
    start_lng DECIMAL(11, 8),
    end_lat DECIMAL(10, 8),
    end_lng DECIMAL(11, 8),
    distance_mi DECIMAL(8, 3),
    description TEXT,
    street VARCHAR(200),
    city VARCHAR(100),
    county VARCHAR(100),
    state VARCHAR(2),
    zipcode VARCHAR(10),
    country VARCHAR(2),
    timezone VARCHAR(50),
    airport_code VARCHAR(10),
    weather_timestamp TIMESTAMP,
    temperature_f DECIMAL(5, 2),
    wind_chill_f DECIMAL(5, 2),
    humidity_pct DECIMAL(5, 2),
    pressure_in DECIMAL(5, 2),
    visibility_mi DECIMAL(5, 2),
    wind_direction VARCHAR(10),
    wind_speed_mph DECIMAL(5, 2),
    precipitation_in DECIMAL(5, 3),
    weather_condition VARCHAR(100),
    amenity BOOLEAN,
    bump BOOLEAN,
    crossing BOOLEAN,
    give_way BOOLEAN,
    junction BOOLEAN,
    no_exit BOOLEAN,
    railway BOOLEAN,
    roundabout BOOLEAN,
    station BOOLEAN,
    stop BOOLEAN,
    traffic_calming BOOLEAN,
    traffic_signal BOOLEAN,
    turning_loop BOOLEAN,
    sunrise_sunset VARCHAR(10),
    civil_twilight VARCHAR(10),
    nautical_twilight VARCHAR(10),
    astronomical_twilight VARCHAR(10)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_accidents_state ON us_accidents(state);
CREATE INDEX IF NOT EXISTS idx_accidents_city ON us_accidents(city);
CREATE INDEX IF NOT EXISTS idx_accidents_start_time ON us_accidents(start_time);
CREATE INDEX IF NOT EXISTS idx_accidents_severity ON us_accidents(severity);
CREATE INDEX IF NOT EXISTS idx_accidents_location ON us_accidents(start_lat, start_lng);