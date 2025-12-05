"""
<<<<<<< HEAD
Computes the route metrics for a route across 1 week.
Finds:
- Total travel time
- Average wait time
- Verhicle-kilometers traveled
- Bus utilization (% of available hours)
=======
Network-level bus route evaluation with date filtering.
>>>>>>> 748e956 (clarke_write metrics and implementation)
"""

import sys
import pandas as pd
import osmnx as ox
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
<<<<<<< HEAD
import os
import pickle

# -------------------------------
# Config
# -------------------------------
FLEET_SIZE = 60
SERVICE_HOURS = 16
AVAILABLE_HOURS = FLEET_SIZE * SERVICE_HOURS
AVERAGE_SPEED_KMH = 25
START_DATE = pd.Timestamp(2021, 11, 18)
END_DATE   = pd.Timestamp(2021, 11, 25)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
GRAPH_CACHE = os.path.join(CACHE_DIR, "osmnx_graph.graphml")
NODES_CACHE = os.path.join(CACHE_DIR, "stop_to_node.pkl")
ROUTE_METRICS_CACHE = os.path.join(CACHE_DIR, "route_metrics.pkl")
=======

# -------------------------------
# Config
# -------------------------------
FLEET_SIZE = 60
SERVICE_HOURS = 16
AVAILABLE_HOURS = FLEET_SIZE * SERVICE_HOURS
AVERAGE_SPEED_KMH = 25
START_DATE = pd.Timestamp(2021, 11, 18)
END_DATE   = pd.Timestamp(2021, 11, 25)

>>>>>>> 748e956 (clarke_write metrics and implementation)

if input(
    'NOTE: This file\'s params MUST be the same as the ones used to generate the routes.' \
    'Please make sure that\'s the case. Confirm y/n.'
) not in ['y', 'Y']:
    sys.exit(0)

# -------------------------------
# File paths
# -------------------------------
<<<<<<< HEAD
args = sys.argv
if len(args) < 3:
    print(f'Usage `{args[0]} <routes.csv> [--dynamic or --static]`')
    sys.exit(1)
else:
    ROUTES_CSV = args[1]
    if args[2] == '--dynamic':
        IS_DYNAMIC = True
    elif args[2] == '--static':
        IS_DYNAMIC = False 
    else:
        print(f'Usage `{args[0]} <routes.csv> [--dynamic or --static]`')
        sys.exit(1)

=======
ROUTES_CSV = "rodes_a/designed_routes/clarke_write-2025-11-18.csv"
>>>>>>> 748e956 (clarke_write metrics and implementation)
STOPS_CSV = "../data/stops.csv"
TRIPS_CSV = "../data/trips.csv"

# -------------------------------
# Filter parameters
# -------------------------------
<<<<<<< HEAD
print(f'Computing metrics for {START_DATE.strftime("%a %d %b %Y")} to {END_DATE.strftime("%a %d %b %Y")}')
=======
START_DATE = pd.Timestamp(2021, 11, 18)
END_DATE   = pd.Timestamp(2021, 11, 18)
>>>>>>> 748e956 (clarke_write metrics and implementation)

# -------------------------------
# Load routes and stops
# -------------------------------
<<<<<<< HEAD
print('Loading routes & stops...')
routes_df = pd.read_csv(ROUTES_CSV)
stops_df = pd.read_csv(STOPS_CSV)
stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops_df.iterrows()}

# -------------------------------
# Build or load OSMnx graph
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
    north = max(lats) + padding
    south = min(lats) - padding
    east = max(lons) + padding
    west = min(lons) - padding

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
=======
routes_df = pd.read_csv(ROUTES_CSV)
stops_df = pd.read_csv(STOPS_CSV)
stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops_df.iterrows()}

# -------------------------------
# Build or load OSMnx graph
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
    north = max(lats) + padding
    south = min(lats) - padding
    east = max(lons) + padding
    west = min(lons) - padding

G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive", simplify=True)
stop_to_node = {sid: ox.nearest_nodes(G, X=lon, Y=lat) for sid, (lat, lon) in stop_lookup.items()}
>>>>>>> 748e956 (clarke_write metrics and implementation)

# -------------------------------
# Precompute route distances & travel times
# -------------------------------
<<<<<<< HEAD
if os.path.exists(ROUTE_METRICS_CACHE):
    print(f'Loading cached route metrics from {ROUTE_METRICS_CACHE}')
    with open(ROUTE_METRICS_CACHE, "rb") as f:
        route_metrics = pickle.load(f)
else:
    print('Precomputing route distances & travel times...')
    route_metrics = {}
    for _, row in tqdm(routes_df.iterrows(), total=len(routes_df)):
        stop_ids = [s.strip() for s in row["stops"].split("→")]
        nodes = [stop_to_node[s] for s in stop_ids]

        km = 0
        for i in range(len(nodes)-1):
            path = nx.shortest_path(G, nodes[i], nodes[i+1], weight="length")
            km += sum(G.edges[path[j], path[j+1], 0]["length"] for j in range(len(path)-1)) / 1000
        route_metrics[row["route_id"]] = {
            "distance_km": km,
            "travel_time_h": km / AVERAGE_SPEED_KMH
        }
    with open(ROUTE_METRICS_CACHE, "wb") as f:
        pickle.dump(route_metrics, f)
=======

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
>>>>>>> 748e956 (clarke_write metrics and implementation)

# -------------------------------
# Process trips in chunks and filter by date
# -------------------------------
<<<<<<< HEAD
print('Processing trips...')
=======
>>>>>>> 748e956 (clarke_write metrics and implementation)
route_trip_counts = defaultdict(int)
chunksize = 10_000_000  # adjust based on memory

for chunk in pd.read_csv(TRIPS_CSV, chunksize=chunksize):
<<<<<<< HEAD
=======
    # Build a datetime column
>>>>>>> 748e956 (clarke_write metrics and implementation)
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
<<<<<<< HEAD
print('Computing metrics...')
total_vehicle_km = sum(route_metrics[r]["distance_km"] * route_trip_counts.get(r, 1)
                       for r in route_metrics)
=======
total_vehicle_km = sum(route_metrics[r]["distance_km"] * route_trip_counts.get(r, 1)
                       for r in route_metrics)

>>>>>>> 748e956 (clarke_write metrics and implementation)
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
