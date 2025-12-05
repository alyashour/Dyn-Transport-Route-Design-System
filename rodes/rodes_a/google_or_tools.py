"""
OR-Tools VRP for large bus network (2000 stops)
Constraints:
- Maximum stops per route: MAX_ROUTE_LENGTH
- Maximum distance per route: MAX_DISTANCE km
Caches the distance matrix to speed up repeated runs.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tqdm import tqdm
import os

# -------------------------------
# Config
# -------------------------------
STOPS_CSV = "../../data/stops.csv"
OUTPUT_CSV = "optimized_routes.csv"
DISTANCE_CACHE = "cache/distance_matrix.npy"

NUM_VEHICLES = 60
DEPOT_STOP_ID = "ADELADA1"
MAX_ROUTE_LENGTH = 25  # max stops per vehicle
MAX_DISTANCE = 30      # max distance per vehicle (km)

# -------------------------------
# Load stops
# -------------------------------
print('Loading stops')
stops_df = pd.read_csv(STOPS_CSV)
stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops_df.iterrows()}
stop_name_lookup = {row["stop_id"]: row["stop_name"] for _, row in stops_df.iterrows()}
stop_ids = list(stop_lookup.keys())
stop_index = {sid: i for i, sid in enumerate(stop_ids)}
n = len(stop_ids)

# -------------------------------
# Load or build distance matrix
# -------------------------------
if os.path.exists(DISTANCE_CACHE):
    print(f'Loading cached distance matrix from {DISTANCE_CACHE}')
    distance_matrix = np.load(DISTANCE_CACHE)
else:
    print('Building distance matrix (Euclidean)')
    distance_matrix = np.zeros((n, n))
    for i, sid_from in tqdm(enumerate(stop_ids), total=n):
        for j, sid_to in enumerate(stop_ids):
            if i == j:
                continue
            distance_matrix[i][j] = geodesic(stop_lookup[sid_from], stop_lookup[sid_to]).km
    print(f'Saving distance matrix to cache: {DISTANCE_CACHE}')
    np.save(DISTANCE_CACHE, distance_matrix)

# -------------------------------
# OR-Tools VRP setup
# -------------------------------
print('Setting up OR-Tools VRP')
manager = pywrapcp.RoutingIndexManager(n, NUM_VEHICLES, stop_index[DEPOT_STOP_ID])
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(distance_matrix[from_node][to_node] * 1000)  # meters

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# -------------------------------
# Add constraints
# -------------------------------
# Max route distance
routing.AddDimension(
    transit_callback_index,
    0,
    int(MAX_DISTANCE * 1000),
    True,
    "Distance"
)
distance_dim = routing.GetDimensionOrDie("Distance")
distance_dim.SetGlobalSpanCostCoefficient(100)

# Max stops per route
def length_callback(from_index, to_index):
    return 1
length_callback_index = routing.RegisterTransitCallback(length_callback)
routing.AddDimension(
    length_callback_index,
    0,
    MAX_ROUTE_LENGTH,
    True,
    "Stops"
)
length_dim = routing.GetDimensionOrDie("Stops")
length_dim.SetGlobalSpanCostCoefficient(10)

# Prevent skipping stops
for i in range(1, n):
    routing.AddDisjunction([manager.NodeToIndex(i)], 1_000_000)

# -------------------------------
# Solve parameters
# -------------------------------
print('Setting solve parameters')
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
search_parameters.time_limit.seconds = 300  # 5 minutes

# -------------------------------
# Solve VRP
# -------------------------------
print('Solving VRP...')
solution = routing.SolveWithParameters(search_parameters)

# -------------------------------
# Output optimized routes to CSV
# -------------------------------
routes_output = []

if solution:
    print('Saving optimized routes')
    for vehicle_id in tqdm(range(NUM_VEHICLES)):
        index = routing.Start(vehicle_id)
        route_stop_ids = []
        route_stop_names = []
        route_distance_km = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            sid = stop_ids[node]
            route_stop_ids.append(sid)
            route_stop_names.append(stop_name_lookup[sid])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance_km += routing.GetArcCostForVehicle(previous_index, index, vehicle_id) / 1000
        if len(route_stop_ids) > 1:
            routes_output.append({
                "route_id": f"Route_{vehicle_id+1}",
                "stops": " → ".join(route_stop_ids),
                "stop_names": " → ".join(route_stop_names),
                "num_stops": len(route_stop_ids),
                "total_distance_km": round(route_distance_km, 2)
            })
else:
    print("No solution found!")

# Save CSV
df_routes = pd.DataFrame(routes_output)
df_routes.to_csv(OUTPUT_CSV, index=False)
print(f"Routes saved to {OUTPUT_CSV}")
