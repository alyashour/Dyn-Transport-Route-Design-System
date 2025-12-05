import pandas as pd
import os

# --- FILE PATHS ---
BIG_DATA_FILE = 'dataset_generator/london_ridership_trip_generation.csv'
STOPS_FILE = 'dataset_generator/stopwithZones.csv'
OUTPUT_FLOW = 'dataset_generator/viz_ready_flows.csv'
OUTPUT_TIME = 'dataset_generator/viz_ready_timeline.csv'

# 1. Load Stop Coordinates
print("Loading stop coordinates...")
stops = pd.read_csv(STOPS_FILE)
# Create a dictionary for fast lookups: ID -> (Lat, Lon)
stops = stops.drop_duplicates(subset=['stop_id'])
stop_map = stops.set_index('stop_id')[['stop_lat', 'stop_lon']].to_dict('index')

# 2. Initialize Aggregators
daily_stats = {}  # Date -> Count
route_stats = {}  # (Origin, Dest) -> Count

# 3. Process the 1.6GB file in chunks
chunk_size = 500000  # Process 500k rows at a time
print("Processing large file in chunks...")

try:
    for chunk in pd.read_csv(BIG_DATA_FILE, chunksize=chunk_size):
        # --- A. Time Aggregation ---
        # Create a date string column for grouping
        chunk['date_str'] = pd.to_datetime(dict(year=chunk.Year, month=chunk.Month, day=chunk.Day)).dt.strftime('%Y-%m-%d')
        
        # Count trips per date in this chunk
        date_counts = chunk['date_str'].value_counts()
        
        # Add to global totals
        for date, count in date_counts.items():
            daily_stats[date] = daily_stats.get(date, 0) + count

        # --- B. Route Aggregation ---
        # Count O-D pairs in this chunk
        route_counts = chunk.groupby(['Origin ID', 'Destination ID']).size()
        
        # Add to global totals
        for (o, d), count in route_counts.items():
            route_stats[(o, d)] = route_stats.get((o, d), 0) + count

    print("Chunk processing complete. formatting outputs...")

    # 4. Save Timeline Data
    timeline_df = pd.DataFrame(list(daily_stats.items()), columns=['Date', 'Trip_Count'])
    timeline_df = timeline_df.sort_values('Date')
    timeline_df.to_csv(OUTPUT_TIME, index=False)
    print(f"-> Created {OUTPUT_TIME}")

    # 5. Save Flow Data (Compatible with Kepler.gl)
    flow_list = []
    for (o_id, d_id), count in route_stats.items():
        # Only keep routes with meaningful volume (e.g., > 5 trips total) to save space
        if count > 5:
            # Look up coordinates
            if o_id in stop_map and d_id in stop_map:
                o_coords = stop_map[o_id]
                d_coords = stop_map[d_id]
                
                flow_list.append({
                    'Origin_ID': o_id,
                    'Dest_ID': d_id,
                    'Origin_Lat': o_coords['stop_lat'],
                    'Origin_Lon': o_coords['stop_lon'],
                    'Dest_Lat': d_coords['stop_lat'],
                    'Dest_Lon': d_coords['stop_lon'],
                    'Trip_Count': count
                })
    
    flow_df = pd.DataFrame(flow_list)
    flow_df.to_csv(OUTPUT_FLOW, index=False)
    print(f"-> Created {OUTPUT_FLOW}")

except FileNotFoundError:
    print("Error: Could not find your CSV files. Check the paths at the top of the script.")