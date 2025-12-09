""" 
Preprocesses the data into a single .h5 file.
Very quick takes like 2 mins on average.
"""

import pandas as pd
import numpy as np
import h5py as hp
import os
import io
from ...util import get_root

# Configuration
ROOT = get_root()
DATA_IN = ROOT / 'data/1_ripr/in'
MEMCACHE = ROOT / 'data/1_ripr/memcache'
# - data in
TRIPS_FILE = DATA_IN / 'trips.csv'
WEATHER_FILE = DATA_IN / 'weather.csv'
STOPS_FILE = DATA_IN / 'stops.csv'
POPULATION_FILE = DATA_IN / 'city_population.csv'
# - data out
OUTPUT_FILE = MEMCACHE / 'dataset.h5'
CHUNK_SIZE = 100000 

def get_weather_index(condition):
    """Maps string condition to uint8 index."""
    match str(condition).lower():
        case s if 'rain' in s:
            return 1
        case s if 'snow' in s:
            return 2
        case _:
            return 0 

def get_csv_date_range(filepath):
    """
    Efficiently determines the start and end dates of a large sorted CSV
    without reading the whole file into RAM.
    """
    print(f" - detecting date range for {os.path.basename(filepath)}...")
    
    # 1. Get Start Date (Read Header + Row 1)
    df_start = pd.read_csv(filepath, nrows=1)
    start_date = pd.to_datetime(df_start[['Year', 'Month', 'Day']]).iloc[0]
    columns = df_start.columns # Save cols for parsing the last line
    
    # 2. Get End Date (Seek to end)
    with open(filepath, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        
        # Buffer to hold the last chunk
        buffer_size = 1024
        # Ensure we don't go past start of file
        seek_pos = max(0, file_size - buffer_size)
        f.seek(seek_pos)
        
        tail_data = f.read()
        
        # Decode and split lines
        lines = tail_data.decode('utf-8', errors='ignore').splitlines()
        
        # Get last non-empty line
        last_line = lines[-1] if lines else ""
        if not last_line.strip(): 
            # If last line is empty (newline at EOF), take the one before
            last_line = lines[-2] if len(lines) > 1 else lines[0]

    # Parse last line using the columns we grabbed earlier
    # We wrap it in StringIO so pandas can parse the CSV format (handling commas etc)
    df_end = pd.read_csv(io.StringIO(last_line), names=columns, header=None)
    end_date = pd.to_datetime(df_end[['Year', 'Month', 'Day']]).iloc[0]
    
    return start_date, end_date

if __name__ == "__main__":
    
    print("Step 1: Processing Stops...")
    stops_df = pd.read_csv(STOPS_FILE)
    stops_df = stops_df.sort_values('stop_id').reset_index(drop=True)
    stop_id_to_idx = {sid: i for i, sid in enumerate(stops_df['stop_id'])}
    num_stops = len(stops_df)
    
    print("Step 2: Syncing Timelines...")
    # 1. Detect range from the massive trips file
    trip_start, trip_end = get_csv_date_range(TRIPS_FILE)
    print(f"   Trips Range: {trip_start.date()} to {trip_end.date()}")

    # 2. Load and CROP weather to match
    weather_df = pd.read_csv(WEATHER_FILE)
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    
    # FILTER: Only keep weather rows within the trips range
    mask = (weather_df['Date'] >= trip_start) & (weather_df['Date'] <= trip_end)
    weather_df = weather_df.loc[mask].sort_values('Date').reset_index(drop=True)
    
    if len(weather_df) == 0:
        raise ValueError("Error: Weather file date range does not overlap with Trips file range!")
    
    # Map Date -> Index (0 to num_days-1)
    # This map is now tightly coupled to the trips range
    date_to_idx = {d.date(): i for i, d in enumerate(weather_df['Date'])}
    num_days = len(weather_df)
    
    years = weather_df['Date'].dt.year.values
    months = weather_df['Date'].dt.month.values
    days = weather_df['Date'].dt.day.values
    
    print(f"   Final Master Timeline: {num_days} days (aligned to trips).")

    # ---------------------------------------------------------
    # INITIALIZE H5 FILE
    # ---------------------------------------------------------
    print("Step 3: Initializing H5 File...")
    
    with hp.File(OUTPUT_FILE, 'w') as f:
        # Static Datasets
        dset_stops = f.create_dataset('stops', shape=(num_stops, 2), dtype='float16')
        dset_stops[:] = stops_df[['stop_lat', 'stop_lon']].astype('float16').values

        pop_df = pd.read_csv(POPULATION_FILE)
        num_years = len(pop_df)
        dset_pop = f.create_dataset('populations', shape=(num_years, 2), dtype='uint32')
        pop_data = np.zeros((num_years, 2), dtype='uint32')
        pop_data[:, 0] = (pop_df['Year'] - 2000).astype('uint32')
        pop_data[:, 1] = pop_df['Population'].astype('uint32')
        dset_pop[:] = pop_data

        # UPDATED DATES DATASET
        # Columns: [Year-2000, Month, Day, DayOfWeek, WeatherIdx]
        # DayOfWeek: 0 = Monday, 6 = Sunday
        dset_dates = f.create_dataset('dates', shape=(num_days, 5), dtype='uint8')
        dates_buffer = np.zeros((num_days, 5), dtype='uint8')
        
        dates_buffer[:, 0] = (years - 2000).astype('uint8')
        dates_buffer[:, 1] = months.astype('uint8')
        dates_buffer[:, 2] = days.astype('uint8')
        dates_buffer[:, 3] = weather_df['Date'].dt.dayofweek.values.astype('uint8')
        dates_buffer[:, 4] = weather_df['Weather_Condition'].apply(get_weather_index).astype('uint8')
        
        dset_dates[:] = dates_buffer

        dset_temps = f.create_dataset('temps', shape=(num_days, 2), dtype='float16')
        dset_temps[:] = weather_df[['temp_high_c', 'temp_low_c']].astype('float16').values

        # UPDATED SUMMARY DATASET
        # Columns: [Total_Count, Start_Idx, End_Idx]
        # Note: We removed Y, M, D from here as they are now in 'dates'
        dset_summary = f.create_dataset('day_trips', shape=(num_days, 3), dtype='uint32')
        
        dset_detailed = f.create_dataset('trips', shape=(0, 4), maxshape=(None, 4), dtype='uint32', chunks=True)

    # ---------------------------------------------------------
    # SINGLE-PASS STREAMING PROCESSING
    # ---------------------------------------------------------
    print("Step 4: Streaming Trips (Sorted Input)...")
    
    # State variables
    current_day_idx = -1
    day_buffer = {} # {(o, d) -> count}
    detailed_write_ptr = 0
    
    chunk_iter = pd.read_csv(TRIPS_FILE, chunksize=CHUNK_SIZE)
    
    def flush_day(d_idx, buffer, write_ptr):
        if not buffer:
            return write_ptr
            
        sorted_keys = sorted(buffer.keys())
        num_rows = len(sorted_keys)
        block = np.zeros((num_rows, 4), dtype='uint32')
        
        total_trips = 0
        for i, (o, d) in enumerate(sorted_keys):
            cnt = buffer[(o, d)]
            block[i, 0] = d_idx
            block[i, 1] = o
            block[i, 2] = d
            block[i, 3] = cnt
            total_trips += cnt
            
        with hp.File(OUTPUT_FILE, 'r+') as f:
            dset_det = f['trips']
            dset_sum = f['day_trips']
            
            dset_det.resize(write_ptr + num_rows, axis=0)
            dset_det[write_ptr : write_ptr + num_rows] = block
            
            # Update summary (Indices 0, 1, 2 now)
            s_data = dset_sum[d_idx]
            s_data[0] = total_trips
            s_data[1] = write_ptr
            s_data[2] = write_ptr + num_rows
            dset_sum[d_idx] = s_data
            
        return write_ptr + num_rows

    total_processed_rows = 0

    for chunk in chunk_iter:
        # 1. Prepare Data
        chunk_dates = pd.to_datetime(chunk[['Year', 'Month', 'Day']]).dt.date
        row_indices = chunk_dates.map(date_to_idx)
        
        # 2. Map Stops
        o_indices = chunk['Origin ID'].map(stop_id_to_idx)
        d_indices = chunk['Destination ID'].map(stop_id_to_idx)
        
        # 3. Filter valid
        # IMPORTANT: row_indices will now be NaN for dates outside the crop range
        mask = row_indices.notna() & o_indices.notna() & d_indices.notna()
        valid_chunk = chunk[mask]
        
        if len(valid_chunk) == 0:
            continue
            
        r_vals = row_indices[mask].astype(int).values
        o_vals = o_indices[mask].astype(int).values
        d_vals = d_indices[mask].astype(int).values
        
        unique_days = np.unique(r_vals)
        
        for d in unique_days:
            if d != current_day_idx:
                if current_day_idx != -1:
                    detailed_write_ptr = flush_day(current_day_idx, day_buffer, detailed_write_ptr)
                    print(f"   Finished Day {current_day_idx} (Total Rows: {detailed_write_ptr})...", end='\r')
                
                current_day_idx = d
                day_buffer = {}
            
            day_mask = (r_vals == d)
            day_o = o_vals[day_mask]
            day_d = d_vals[day_mask]
            
            local_df = pd.DataFrame({'o': day_o, 'd': day_d})
            local_agg = local_df.groupby(['o', 'd']).size().reset_index(name='cnt')
            
            for row in local_agg.itertuples(index=False):
                key = (row.o, row.d)
                day_buffer[key] = day_buffer.get(key, 0) + row.cnt
                
        total_processed_rows += len(chunk)

    if current_day_idx != -1:
        detailed_write_ptr = flush_day(current_day_idx, day_buffer, detailed_write_ptr)

    print(f"\nSuccess! Created {OUTPUT_FILE}")
    print(f" - Detailed Entries: {detailed_write_ptr}")
    print(f" - Daily Summary Rows: {num_days}")
