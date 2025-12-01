import pandas as pd
from collections import defaultdict

def build_demand_graph(trips_csv_path: str):
    """
    Build an Origin-Destination demand graph from the trip CSV.

    """

    df = pd.read_csv(trips_csv_path)

    df.columns = [c.strip().lower() for c in df.columns]

   
    origin_col = "origin id"
    dest_col   = "destination id"

    demand_graph = defaultdict(lambda: defaultdict(int))

    # Aggregate all trips O -> D
    for _, row in df.iterrows():
        o = row[origin_col]
        d = row[dest_col]
        demand_graph[o][d] += 1

    # Extract all unique stop IDs
    stop_ids = sorted(set(df[origin_col]).union(set(df[dest_col])))

    return demand_graph, stop_ids
