import numpy as np
import pandas as pd

from route_designer.clarke_wright import clarke_wright_routes
from route_designer.cost_matrix import build_cost_matrix
from RIPR_B.RIPR_B_predict import predict_total_riders

# --- USER SETTINGS ---
GRID_FILE = "ridr_a/predictions/grid_2021-11-18.npy"  # 1 grid
STOPS_CSV = "dataset_generator/stopwithZones.csv"
DEPOT_ID = "ADELCEN1"
DATE = "2021-11-18"
VEHICLE_CAPACITY = 60


# ---- STEP 1: Load RIPR-A Grid (probabilities) ----
print("Loading RIPR-A grid...")
grid = np.load(GRID_FILE)  # n x n matrix


# ---- STEP 2: Get RIPR-B Total Riders (scalar) ----
print("Predicting total riders...")
N = predict_total_riders(DATE, weather_label="clear", temp=10)

print(f"Predicted riders = {N}")


# ---- STEP 3: Multiply to get O-D Matrix ----
print("Computing demand matrix...")
demand_matrix = grid * N


# ---- STEP 4: Convert to demand_graph ----
print("Loading stops...")
stops = pd.read_csv(STOPS_CSV)
stop_ids = stops["stop_id"].tolist()

print("Converting to graph...")
demand_graph = {}
n = len(stop_ids)

for i in range(n):
    origin = stop_ids[i]
    demand_graph[origin] = {}
    for j in range(n):
        dest = stop_ids[j]
        demand_graph[origin][dest] = float(demand_matrix[i, j])


# ---- STEP 5: Build cost matrix ----
print("Building cost matrix...")
cost_matrix, stop_ids = build_cost_matrix(STOPS_CSV)


# ---- STEP 6: Run RODES ----
print("Running RODES...")
routes = clarke_wright_routes(
    demand_graph=demand_graph,
    cost_matrix=cost_matrix,
    stop_ids=stop_ids,
    depot_id=DEPOT_ID,
    vehicle_capacity=VEHICLE_CAPACITY
)


# ---- STEP 7: Print ----
print("\nRoutes:")
for i, r in enumerate(routes, start=1):
    print(f"{i}: {r}")
