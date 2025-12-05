"""
Network-level bus route evaluation with date filtering.
"""

import sys
import pandas as pd
import osmnx as ox
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

# Bus utilization
FLEET_SIZE = 60
SERVICE_HOURS = 16
AVAILABLE_HOURS = FLEET_SIZE * SERVICE_HOURS
AVERAGE_SPEED_KMH = 25  # average bus speed


if input(
    'NOTE: This file\'s params MUST be the same as the ones used to generate the routes.' \
    'Please make sure that\'s the case. Confirm y/n.'
) not in ['y', 'Y']:
    sys.exit(0)

# -------------------------------
# File paths
# -------------------------------
ROUTES_CSV = "rodes_a/designed_routes/clarke_write-2025-11-18.csv"
STOPS_CSV = "../data/stops.csv"
TRIPS_CSV = "../data/trips.csv"

# -------------------------------
# Filter parameters
# -------------------------------
START_DATE = pd.Timestamp(2021, 11, 18)
END_DATE   = pd.Timestamp(2021, 11, 18)

# -------------------------------
# Load routes and stops
# -------------------------------
routes_df = pd.read_csv(ROUTES_CSV)
stops_df = pd.read_csv(STOPS_CSV)

stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops_df.iterrows()}

# -------------------------------
# Build OSMnx graph
# -------------------------------
all_coords = list(stop_lookup.values())
lats = [c[0] for c in all_coords]
lons = [c[1] for c in all_coords]

padding = 0.002
north = max(lats) + padding
south = min(lats) - padding
east = max(lons) + padding
west = min(lons) - padding

G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive", simplify=True)
stop_to_node = {sid: ox.nearest_nodes(G, X=lon, Y=lat) for sid, (lat, lon) in stop_lookup.items()}

# -------------------------------
# Precompute route distances & travel times
# -------------------------------

route_metrics = {}
for _, row in routes_df.iterrows():
    stop_ids = [s.strip() for s in row["stops"].split("→")]
    nodes = [stop_to_node[s] for s in stop_ids]

    km = 0
    for i in range(len(nodes)-1):
        path = nx.shortest_path(G, nodes[i], nodes[i+1], weight="length")
        km += sum(G.edges[path[j], path[j+1], 0]["length"] for j in range(len(path)-1)) / 1000
    route_metrics[row["route_id"]] = {"distance_km": km,
                                     "travel_time_h": km / AVERAGE_SPEED_KMH}

# -------------------------------
# Process trips in chunks and filter by date
# -------------------------------
route_trip_counts = defaultdict(int)
chunksize = 10_000_000  # adjust based on memory

for chunk in pd.read_csv(TRIPS_CSV, chunksize=chunksize):
    # Build a datetime column
    chunk["date"] = pd.to_datetime(chunk[["Year","Month","Day"]])
    chunk = chunk[(chunk["date"] >= START_DATE) & (chunk["date"] <= END_DATE)]

    for _, trip in chunk.iterrows():
        origin, dest = trip["Origin ID"], trip["Destination ID"]
        for _, row in routes_df.iterrows():
            stop_ids = [s.strip() for s in row["stops"].split("→")]
            if origin in stop_ids and dest in stop_ids:
                route_trip_counts[row["route_id"]] += 1
                break

# -------------------------------
# Compute network-level metrics
# -------------------------------
total_vehicle_km = sum(route_metrics[r]["distance_km"] * route_trip_counts.get(r, 1)
                       for r in route_metrics)

total_travel_time_h = sum(route_metrics[r]["travel_time_h"] * route_trip_counts.get(r, 1)
                          for r in route_metrics)

# Average wait time (approx: inverse of trips per hour)
avg_wait_time_h = 0
for route_id, trips in tqdm(route_trip_counts.items()):
    frequency_per_h = trips / 24  # trips per hour for that day
    if frequency_per_h > 0:
        avg_wait_time_h += 1 / frequency_per_h
avg_wait_time_h /= len(route_trip_counts) if route_trip_counts else 1

# Bus utilization
bus_utilization_pct = total_travel_time_h / AVAILABLE_HOURS * 100

# -------------------------------
# Print metrics
# -------------------------------
print("=== Network-level Metrics ===")
print(f"Total travel time (h): {total_travel_time_h:.2f}")
print(f"Average wait time (h): {avg_wait_time_h:.2f}")
print(f"Vehicle-kilometers traveled (km): {total_vehicle_km:.2f}")
print(f"Bus utilization (% of available hours): {bus_utilization_pct:.1f}%")
