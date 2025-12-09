"""
This converts GTFS data describing London's bus routes
into the route.csv files our scripts use.
"""
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm

# -------------------------------
# File paths
# -------------------------------
ROUTES_TXT = "../data/google_transit/routes.txt"
TRIPS_TXT = "../data/google_transit/trips.txt"
STOP_TIMES_TXT = "../data/google_transit/stop_times.txt"
STOPS_TXT = "../data/google_transit/stops.txt"
SHAPES_TXT = "../data/google_transit/shapes.txt"
OUTPUT_CSV = "gtfs_routes.csv"

# -------------------------------
# Load GTFS data
# -------------------------------
routes = pd.read_csv(ROUTES_TXT)
trips = pd.read_csv(TRIPS_TXT)
stop_times = pd.read_csv(STOP_TIMES_TXT)
stops = pd.read_csv(STOPS_TXT)
shapes = pd.read_csv(SHAPES_TXT)

stop_lookup = {row['stop_id']: (row['stop_lat'], row['stop_lon']) for _, row in stops.iterrows()}
stop_name_lookup = {row['stop_id']: row['stop_name'] for _, row in stops.iterrows()}

# -------------------------------
# Reconstruct each route
# -------------------------------
routes_output = []

for idx, route_row in tqdm(routes.iterrows(), total=len(routes)):

    route_id = route_row['route_id']

    # Pick one trip from this route
    route_trips = trips[trips['route_id'] == route_id]
    if route_trips.empty:
        continue
    trip_id = route_trips.iloc[0]['trip_id']

    # Get stop sequence for this trip
    trip_stop_times = stop_times[stop_times['trip_id'] == trip_id].sort_values('stop_sequence')
    stop_ids = trip_stop_times['stop_id'].tolist()
    stop_names = [stop_name_lookup[sid] for sid in stop_ids]

    # Estimate total distance along shape
    shape_id = route_trips.iloc[0]['shape_id'] if 'shape_id' in route_trips.columns else None
    total_distance_km = 0
    if shape_id is not None:
        shape_points = shapes[shapes['shape_id'] == shape_id].sort_values('shape_pt_sequence')
        coords = list(zip(shape_points['shape_pt_lat'], shape_points['shape_pt_lon']))
        for i in range(len(coords) - 1):
            total_distance_km += geodesic(coords[i], coords[i+1]).km
    else:
        # fallback: sum distance between stops
        coords = [stop_lookup[sid] for sid in stop_ids]
        for i in range(len(coords) - 1):
            total_distance_km += geodesic(coords[i], coords[i+1]).km

    routes_output.append({
        'route_id': f"Route_{idx+1}", # type: ignore
        'stops': " → ".join(stop_ids),
        'stop_names': " → ".join(stop_names),
        'num_stops': len(stop_ids),
        'total_distance_km': round(total_distance_km, 2)
    })

# -------------------------------
# Save CSV
# -------------------------------
df_routes = pd.DataFrame(routes_output)
df_routes.to_csv(OUTPUT_CSV, index=False)
print(f"Routes saved to {OUTPUT_CSV}")
