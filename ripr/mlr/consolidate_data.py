import pandas as pd
from typing import Dict, Tuple
import os

from transit_data_processor import TransitDataProcessor

# --- Configuration ---
TRIPS_CSV = 'data/trips_data.csv'
WEATHER_CSV = 'data/weather.csv'
STOPS_CSV = 'data/stop_data.csv'
POPULATION_CSV = 'data/city_population.csv'
OUTPUT_CSV = 'data/processed_combined_data.csv'

def consolidate_and_engineer_data():
    """
    Loads all raw CSV files, performs feature engineering, and saves the 
    aggregated daily data to a single output CSV.
    """
    print(f"--- Starting Data Processing and Engineering ---")
    
    # 1. Initialize Processor and Load Data
    # Assuming TransitDataProcessor handles initialization of mappings (stop_id_to_idx, etc.)
    processor = TransitDataProcessor()
    
    # Check if input files exist before proceeding
    if not all(os.path.exists(f) for f in [TRIPS_CSV, WEATHER_CSV, STOPS_CSV, POPULATION_CSV]):
        print("Error: One or more input CSV files not found. Check paths.")
        return

    processor.load_data(TRIPS_CSV, WEATHER_CSV, STOPS_CSV, POPULATION_CSV)
    print("Raw data loaded.")
    
    # 2. Engineer Features
    # This step adds time features, distance, and stop indices to the trip-level data.
    # It also handles merging weather and population data.
    processor.engineer_features()
    print("Feature engineering complete.")

    # 3. Aggregate to Daily Route Level
    # This step groups the trip data by date and route, calculating the
    # trip_count, total_trips, and the critical route_percentage target variable.
    processor.aggregate_daily_data()
    print("Data aggregated.")
    
    # The final DataFrame is stored in processor.daily_data
    final_df = processor.daily_data
    
    # 4. Save the Consolidated Data
    
    # Optional: Display the final structure and size
    print("\n| Final Consolidated DataFrame")
    print(f"Shape: {final_df.shape}")
    print(f"Columns:\n{final_df.columns.tolist()}")
    
    # Save the DataFrame to the specified output CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    consolidate_and_engineer_data()