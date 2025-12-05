"""
Weekly network-level bus route evaluation for static and dynamic routes.
Now includes:
    - Route-level utilization (% full)
    - Network-level average utilization

Usage:
    python measure_route_metrics.py <routes_path> [--static | --dynamic]
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
VEHICLE_CAPACITY = 50

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
GRAPH_CACHE = os.path.join(CACHE_DIR, "osmnx_graph.graphml")
NODES_CACHE = os.path.join(CACHE_DIR, "stop_to_node.pkl")

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

START_DATE = pd.Timestamp(2021, 11, 18)
END_DATE   = pd.Timestamp(2021, 11, 24)
print(f'Computing metrics for {START_DATE} → {END_DATE}')

# -------------------------------
# Load stops
# -------------------------------
stops_df = pd.read_csv(STOPS_CSV)
stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops_df.iterrows()}

# -------------------------------
# Load or build OSMnx graph
# -------------------------------
if os.path.exists(GRAPH_CACHE):
    G = ox.load_graphml(GRAPH_CACHE)
else:
    all_coords = list(stop_lookup.values())
    lats = [c[0] for c in all_coords]; lons = [c[1] for c in all_coords]
    pad = 0.002
    north, south = max(lats)+pad, min(lats)-pad
    east, west = max(lons)+pad, min(lons)-pad
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive", simplify=True)
    ox.save_graphml(G, GRAPH_CACHE)

# -------------------------------
# Precompute nearest nodes
# -------------------------------
if os.path.exists(NODES_CACHE):
    with open(NODES_CACHE, "rb") as f:
        stop_to_node = pickle.load(f)
else:
    stop_to_node = {sid: ox.nearest_nodes(G, X=lon, Y=lat) for sid, (lat,lon) in stop_lookup.items()}
    with open(NODES_CACHE, "wb") as f:
        pickle.dump(stop_to_node, f)

# -------------------------------
# Compute route metrics
# -------------------------------
def compute_route_metrics(routes_df):
    metrics = {}
    for _, row in tqdm(routes_df.iterrows(), total=len(routes_df)):
        stop_ids = [s.strip() for s in row["stops"].split("→")]
        nodes = [stop_to_node[s] for s in stop_ids]
        km = 0
        for i in range(len(nodes)-1):
            path = nx.shortest_path(G, nodes[i], nodes[i+1], weight="length")
            km += sum(G.edges[path[j], path[j+1],0]["length"]
                      for j in range(len(path)-1)) / 1000
        metrics[row["route_id"]] = {
            "distance_km": km,
            "travel_time_h": km / AVERAGE_SPEED_KMH
        }
    return metrics

# -------------------------------
# Load routes (static or dynamic)
# -------------------------------
if IS_DYNAMIC:
    dates = pd.date_range(START_DATE, END_DATE)
    daily_route_files = [ROUTES_PATH.format(date=d.strftime("%Y_%m_%d")) for d in dates]
else:
    daily_route_files = [ROUTES_PATH]

# -------------------------------
# Process trips
# -------------------------------
route_trip_counts_week = defaultdict(int)
route_metrics_week = {}

chunksize = 10_000_000

for route_file in daily_route_files:
    routes_df = pd.read_csv(route_file)

    # Compute or load metrics
    metrics = compute_route_metrics(routes_df)

    # Merge metrics
    for k,v in metrics.items():
        route_metrics_week[k] = v

    # Count trips
    for chunk in pd.read_csv(TRIPS_CSV, chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk[["Year","Month","Day"]])
        chunk = chunk[(chunk["date"]>=START_DATE)&(chunk["date"]<=END_DATE)]
        for _, trip in chunk.iterrows():
            o, d = trip["Origin ID"], trip["Destination ID"]
            for _, row in routes_df.iterrows():
                stops = [s.strip() for s in row["stops"].split("→")]
                if o in stops and d in stops:
                    route_trip_counts_week[row["route_id"]] += 1
                    break

# -------------------------------
# Compute network metrics
# -------------------------------
total_vehicle_km = 0
total_travel_time_h = 0

### ADDED: route-level utilization results
route_utilization = {}

for route_id, metrics in route_metrics_week.items():
    km = metrics["distance_km"]
    hours = metrics["travel_time_h"]
    trips = route_trip_counts_week.get(route_id, 0)

    total_vehicle_km += km * max(trips, 1)
    total_travel_time_h += hours * max(trips, 1)

    ### ADDED — Compute % full for this route
    if hours > 0:
        passengers_per_hour = trips / hours
        utilization = (passengers_per_hour / VEHICLE_CAPACITY) * 100
    else:
        utilization = 0

    route_utilization[route_id] = utilization

# avg wait time
avg_wait_time_h = 0
for route_id, trips in route_trip_counts_week.items():
    freq_per_h = trips / (24*7)
    if freq_per_h > 0:
        avg_wait_time_h += 1 / freq_per_h
avg_wait_time_h /= max(len(route_trip_counts_week), 1)

# bus availability utilization
bus_utilization_pct = total_travel_time_h / AVAILABLE_HOURS * 100

# -------------------------------
# Print metrics
# -------------------------------
print("=== Weekly Network Metrics ===")
print(f"Total travel time (h): {total_travel_time_h:.2f}")
print(f"Average wait time (h): {avg_wait_time_h:.2f}")
print(f"Vehicle-km traveled: {total_vehicle_km:.1f} km")
print(f"Bus utilization (% hours used): {bus_utilization_pct:.1f}%")

print("\n=== Route Utilization (% full) ===")
for r, u in sorted(route_utilization.items(), key=lambda x: -x[1]):
    print(f"{r:20s}  {u:6.2f}% full")

