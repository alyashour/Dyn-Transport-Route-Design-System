import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from transit_data_processor import TransitDataProcessor
from util import *
from mlr_grid import MLRGrid

# Assume TransitDataProcessor, MLRGrid, haversine_distance, 
# get_season, is_holiday, and other necessary imports 
# (like LinearRegression) are available in the scope where this script runs.

# --- Helper Function to Create True Grid ---

def create_true_grid(processor: TransitDataProcessor, target_date: str) -> np.ndarray:
    """
    Computes the true route percentage grid for a single specified date.
    
    Args:
        processor: The initialized TransitDataProcessor instance.
        target_date: Date string (e.g., '2023-10-26').
        
    Returns:
        A NumPy array of shape (N_stops, N_stops) with true percentages.
    """
    date = pd.to_datetime(target_date)
    
    # Filter the aggregated trips data for the specific date
    day_trips = processor.trips[
        (processor.trips['Year'] == date.year) &
        (processor.trips['Month'] == date.month) &
        (processor.trips['Day'] == date.day)
    ].copy()

    if day_trips.empty:
        print(f"No trip data found for the date {target_date}. Returning empty grid.")
        return np.zeros((processor.n_stops, processor.n_stops))
    
    # 1. Count total trips for the day
    total_trips = len(day_trips)
    
    # 2. Count trips per route
    route_counts = day_trips.groupby(['origin_idx', 'dest_idx']).size().reset_index(name='trip_count')
    
    # 3. Calculate percentage
    route_counts['route_percentage'] = route_counts['trip_count'] / total_trips
    
    # 4. Populate the grid
    true_grid = np.zeros((processor.n_stops, processor.n_stops), dtype=np.float32)
    
    for _, row in route_counts.iterrows():
        origin = int(row['origin_idx'])
        dest = int(row['dest_idx'])
        true_grid[origin, dest] = row['route_percentage']
        
    return true_grid

# --- Main Script Function ---

def visualize_grids(target_date: str, model_path: str):
    """
    Loads data and model, computes true and predicted grids for a date, 
    and saves them as normalized grayscale images.
    """
    
    # --- Configuration ---
    TRIPS_CSV = 'data/trips_data_small.csv'
    WEATHER_CSV = 'data/weather.csv'
    STOPS_CSV = 'data/stop_data.csv'
    POPULATION_CSV = 'data/city_population.csv'
    
    print(f"--- Processing Date: {target_date} ---")
    
    # 1. Load and Preprocess Data (CPU bound)
    processor = TransitDataProcessor()
    processor.load_data(TRIPS_CSV, WEATHER_CSV, STOPS_CSV, POPULATION_CSV)
    
    # NOTE: We must ensure the scaler is fitted, 
    # so we run the full feature engineering and aggregate steps.
    processor.engineer_features()
    processor.aggregate_daily_data()
    
    # Need to split and fit scaler on train data to make predictions accurate
    # For visualization, we will fit the scaler on the whole dataset for simplicity
    # but in real-world use, this should be done on training data.
    X_all = processor.prepare_features(processor.daily_data)
    processor.scaler.fit(X_all) 
    
    N_STOPS = processor.n_stops
    print(f"Grid size: {N_STOPS}x{N_STOPS}")

    # 2. Load Model
    try:
        mlr_grid = MLRGrid.load(model_path)
        print(f"Successfully loaded MLR Grid model from {model_path}.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Exiting.")
        return

    # 3. Create True Grid (Ground Truth)
    true_grid = create_true_grid(processor, target_date)
    
    # 4. Create Prediction Grid (Model Output)
    
    # We need a small DataFrame for the target date to feed into predict
    target_date_dt = pd.to_datetime(target_date)
    daily_data_row = processor.daily_data[
        (processor.daily_data['Year'] == target_date_dt.year) &
        (processor.daily_data['Month'] == target_date_dt.month) &
        (processor.daily_data['Day'] == target_date_dt.day)
    ]

    if daily_data_row.empty:
        print(f"Error: Could not find aggregated features for {target_date}. Cannot predict.")
        pred_grid = np.zeros((N_STOPS, N_STOPS))
    else:
        # The MLRGrid.predict method expects a DataFrame containing the day's features 
        # and route indices, but it internally groups by date. We can safely pass 
        # the filtered data if it contains the necessary rows.
        
        # get rid of output var (no data leaks allowed lil bro)
        data_for_prediction = daily_data_row.drop(columns=['route_percentage'], errors='ignore')

        print(f'Prediction Data:\n{data_for_prediction.columns}')

        # NOTE: The predict method returns a list of prediction results.
        # We only expect one result for our target_date.
        pred_results = mlr_grid.predict(data_for_prediction, processor.scaler)
        
        if pred_results:
            pred_grid = pred_results[0]['matrix']
        else:
            print("Warning: Model returned no predictions. Using empty grid.")
            pred_grid = np.zeros((N_STOPS, N_STOPS))


    # 5. Normalize and Save Grids as Images
    
    # --- Normalization ---
    # The requirement is: black is the highest value (max percentage)
    # matplotlib's 'gray' colormap: 0.0=Black, 1.0=White
    # To meet the requirement, we use 'gray_r' (reversed gray), 
    # where 1.0=Black, 0.0=White.
    
    # Find the global maximum across both grids for consistent scaling
    global_max = max(true_grid.max(), pred_grid.max())
    if global_max == 0:
        global_max = 1.0 # Avoid division by zero if grids are empty
        
    print(f"Global max percentage (used for normalization): {global_max:.4f}")

    # Function to save a single grid
    def save_grid_image(grid, filename, global_max):
        """Saves a grid as a normalized grayscale bitmap image."""
        
        # Use the 99.5th percentile of non-zero values for visualization contrast
        non_zero_values = grid[grid > 0]
        if non_zero_values.size > 0:
            # Use a high percentile of the grid's own data for better contrast
            v_max_visual = np.percentile(non_zero_values, 99.5) 
        else:
            v_max_visual = global_max
        
        plt.figure(figsize=(10, 10))
        
        # Use cmap='gray_r' (reversed gray: max value is black)
        # Use vmin/vmax to control the contrast mapping
        plt.imshow(grid, cmap='gray_r', vmin=0, vmax=v_max_visual, interpolation='none') 
        
        plt.title(f"{filename.replace('.png', '').upper()} Grid for {target_date}")
        plt.colorbar(label='Trip Percentage Chance')
        plt.xlabel('Destination Stop Index')
        plt.ylabel('Origin Stop Index')
        
        # --- CHANGE HERE: Save as BMP ---
        plt.savefig(filename, format='png')
        plt.close()
        print(f"Saved: {filename}")

    # Find the global maximum across both grids for consistent scaling
    global_max = 0.001 # Start with a small, reasonable max for initialization
    if true_grid.max() > global_max:
        global_max = true_grid.max()
    if pred_grid.max() > global_max:
        global_max = pred_grid.max()

    print(f"Global max percentage (used for normalization): {global_max:.4f}")

    # Save the images
    true_filename = f"images/true_grid_{target_date}.png"
    pred_filename = f"images/model_grid_{target_date}.png"
    
    save_grid_image(true_grid, true_filename, global_max)
    save_grid_image(pred_grid, pred_filename, global_max)
    
    print("--- Visualization Complete ---")


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_model.py /path/to/model.pkl")
        sys.exit(1)

    model_path = sys.argv[1]

    target_date_input = input("Enter a target date (YYYY-MM-DD), e.g., 2024-01-15: ")

    try:
        datetime.strptime(target_date_input, '%Y-%m-%d') # Validate format
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")

    visualize_grids(target_date_input, model_path)