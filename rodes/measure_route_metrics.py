"""
Weekly network-level bus route evaluation for static and dynamic routes.

Usage:
    python measure_route_metrics.py <routes_path> [--static | --dynamic]

Example:
    python measure_route_metrics.py "rodes_a/designed_routes/clarke_write-{date}.csv" --dynamic

Static: single routes.csv applies for the whole week
Dynamic: multiple {model}-route-{date}.csv files for each day
"""

import sys
import pandas as pd
import osmnx as ox
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import os
import pickle

# -------------------------------
# Config
# -------------------------------
FLEET_SIZE = 60
SERVICE_HOURS = 16
AVAILABLE_HOURS = FLEET_SIZE * SERVICE_HOURS
AVERAGE_SPEED_KMH = 25

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
GRAPH_CACHE = os.path.join(CACHE_DIR, "osmnx_graph.graphml")
NODES_CACHE = os.path.join(CACHE_DIR, "stop_to_node.pkl")
ROUTE_METRICS_CACHE = os.path.join(CACHE_DIR, "route_metrics.pkl")

# -------------------------------
# CLI arguments
# -------------------------------
args = sys.argv
if len(args) < 3:
    print(f'Usage `{args[0]} <routes_path> [--static | --dynamic]`')
    sys.exit(1)
ROUTES_PATH = args[1]
IS_DYNAMIC = args[2] == '--dynamic'

# -------------------------------
# File paths
# -------------------------------
STOPS_CSV = "../data/stops.csv"
TRIPS_CSV = "../data/trips.csv"

# -------------------------------
# Weekly filter parameters
# -------------------------------
START_DATE = pd.Timestamp(2021, 11, 18)
END_DATE   = pd.Timestamp(2021, 11, 24)
print(f'Computing metrics for {START_DATE.strftime("%a %d %b %Y")} to {END_DATE.strftime("%a %d %b %Y")}')

# -------------------------------
# Load stops
# -------------------------------
print('Loading stops...')
stops_df = pd.read_csv(STOPS_CSV)
stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops_df.iterrows()}

# -------------------------------
# Load or build OSMnx graph
# -------------------------------
if os.path.exists(GRAPH_CACHE):
    print(f'Loading cached OSMnx graph from {GRAPH_CACHE}')
    G = ox.load_graphml(GRAPH_CACHE)
else:
    print('Building OSMnx graph...')
    all_coords = list(stop_lookup.values())
    lats = [c[0] for c in all_coords]
    lons = [c[1] for c in all_coords]
    padding = 0.002
    north, south = max(lats)+padding, min(lats)-padding
    east, west = max(lons)+padding, min(lons)-padding
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive", simplify=True)
    ox.save_graphml(G, GRAPH_CACHE)

# -------------------------------
# Precompute nearest nodes
# -------------------------------
if os.path.exists(NODES_CACHE):
    print(f'Loading cached nearest nodes from {NODES_CACHE}')
    with open(NODES_CACHE, "rb") as f:
        stop_to_node = pickle.load(f)
else:
    print('Precomputing nearest nodes...')
    stop_to_node = {sid: ox.nearest_nodes(G, X=lon, Y=lat) for sid, (lat, lon) in stop_lookup.items()}
    with open(NODES_CACHE, "wb") as f:
        pickle.dump(stop_to_node, f)

# -------------------------------
# Helper: compute route metrics
# -------------------------------
def compute_route_metrics(routes_df):
    metrics = {}
    for _, row in tqdm(routes_df.iterrows(), total=len(routes_df)):
        stop_ids = [s.strip() for s in row["stops"].split("→")]
        nodes = [stop_to_node[s] for s in stop_ids]
        km = 0
        for i in range(len(nodes)-1):
            path = nx.shortest_path(G, nodes[i], nodes[i+1], weight="length")
            km += sum(G.edges[path[j], path[j+1], 0]["length"] for j in range(len(path)-1)) / 1000
        metrics[row["route_id"]] = {
            "distance_km": km,
            "travel_time_h": km / AVERAGE_SPEED_KMH
        }
    return metrics

# -------------------------------
# Load routes (static or dynamic)
# -------------------------------
if IS_DYNAMIC:
    # dynamic: multiple files, one per day
    date_range = pd.date_range(START_DATE, END_DATE)
    daily_routes_files = [ROUTES_PATH.format(date=d.strftime("%Y_%m_%d")) for d in date_range]
else:
    daily_routes_files = [ROUTES_PATH]

# -------------------------------
# Process trips in chunks & compute cumulative metrics
# -------------------------------
route_trip_counts_week = defaultdict(int)
route_metrics_week = {}

chunksize = 10_000_000

for route_file in daily_routes_files:
    print(f'Processing route file: {route_file}')
    routes_df = pd.read_csv(route_file)

    # Compute route metrics (cached per file to avoid recompute)
    cache_file = os.path.join(CACHE_DIR, f'route_metrics_{os.path.basename(route_file)}.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            route_metrics = pickle.load(f)
    else:
        route_metrics = compute_route_metrics(routes_df)
        with open(cache_file, "wb") as f:
            pickle.dump(route_metrics, f)

    # Merge metrics into cumulative week metrics
    for k, v in route_metrics.items():
        route_metrics_week[k] = v

    # Process trips
    for chunk in pd.read_csv(TRIPS_CSV, chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk[["Year","Month","Day"]])
        chunk = chunk[(chunk["date"] >= START_DATE) & (chunk["date"] <= END_DATE)]
        for _, trip in chunk.iterrows():
            origin, dest = trip["Origin ID"], trip["Destination ID"]
            for _, row in routes_df.iterrows():
                stop_ids = [s.strip() for s in row["stops"].split("→")]
                if origin in stop_ids and dest in stop_ids:
                    route_trip_counts_week[row["route_id"]] += 1
                    break

# -------------------------------
# Compute network-level metrics
# -------------------------------
print('Computing weekly metrics...')
total_vehicle_km = sum(route_metrics_week[r]["distance_km"] * route_trip_counts_week.get(r, 1)
                       for r in route_metrics_week)
total_travel_time_h = sum(route_metrics_week[r]["travel_time_h"] * route_trip_counts_week.get(r, 1)
                          for r in route_metrics_week)

# Average wait time
avg_wait_time_h = 0
for route_id, trips in tqdm(route_trip_counts_week.items()):
    frequency_per_h = trips / (24 * 7)  # trips per hour for the week
    if frequency_per_h > 0:
        avg_wait_time_h += 1 / frequency_per_h
avg_wait_time_h /= len(route_trip_counts_week) if route_trip_counts_week else 1

# Bus utilization
bus_utilization_pct = total_travel_time_h / AVAILABLE_HOURS * 100

# -------------------------------
# Print metrics
# -------------------------------
print("=== Weekly Network-level Metrics ===")
print(f"Total travel time (h): {total_travel_time_h:.2f}")
print(f"Average wait time (h): {avg_wait_time_h:.2f}")
print(f"Vehicle-kilometers traveled (km): {total_vehicle_km:.2f}")
print(f"Bus utilization (% of available hours): {bus_utilization_pct:.1f}%")
