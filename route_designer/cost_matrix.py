import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the Haversine distance between two lat/lon points.
    Returns distance in kilometers.
    """
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def build_cost_matrix(stops_csv_path: str):
    """
    Build a 2D cost matrix of distances between stops.

    Input:
        stops_csv_path: path to CSV containing stop_id, stop_lat, stop_lon

    Output:
        cost_matrix: 2D numpy array
        stop_ids: list of stop IDs in correct matrix order
    """
    df = pd.read_csv(stops_csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Extract required data
    stop_ids = df["stop_id"].tolist()
    lats = df["stop_lat"].tolist()
    lons = df["stop_lon"].tolist()

    n = len(stop_ids)
    cost_matrix = np.zeros((n, n))

    # Compute distances
    for i in range(n):
        for j in range(n):
            cost_matrix[i][j] = haversine_distance(
                lats[i], lons[i],
                lats[j], lons[j]
            )

    return cost_matrix, stop_ids
