"""
High-level orchestration for RODES.

This ties together:
- demand_graph.build_demand_graph(...)
- cost_matrix.build_cost_matrix(...)
- clarke_wright.clarke_wright_routes(...)
"""

import json
from route_designer.demand_graph import build_demand_graph
from route_designer.cost_matrix import build_cost_matrix
from route_designer.clarke_wright import clarke_wright_routes


def design_routes(
    trips_csv_path: str,
    stops_csv_path: str,
    depot_id: int,
    vehicle_capacity: int = 60,
    save_path: str = None
):
    """
    End-to-end route design pipeline.

    Inputs:
        trips_csv_path: path to trip CSV with Origin/Destination IDs.
        stops_csv_path: path to stop info CSV with coordinates.
        depot_id: stop ID of the central depot.
        vehicle_capacity: bus capacity.
        save_path: optional JSON save path.

    Returns:
        routes: list of bus routes (each is a list of stop IDs)
    """

    print("Loading demand graph...")
    demand_graph, stop_ids = build_demand_graph(trips_csv_path)

    print("Building cost matrix...")
    cost_matrix, stop_ids = build_cost_matrix(stops_csv_path)

    print("Running Clarke–Wright Route Optimization...")
    routes = clarke_wright_routes(
        demand_graph=demand_graph,
        cost_matrix=cost_matrix,
        stop_ids=stop_ids,
        depot_id=depot_id,
        vehicle_capacity=vehicle_capacity,
    )

    print("\n✅ Route design finished!")
    for i, r in enumerate(routes, start=1):
        print(f"Route {i}: {r}")

    if save_path:
        print(f"💾 Saving routes to {save_path} ...")
        with open(save_path, "w") as f:
            json.dump(routes, f, indent=4)
        print("✔ Routes saved.")

    return routes


# Example usage when run directly:
if __name__ == "__main__":
    design_routes(
        trips_csv_path="dataset_generator/london_ridership_trip_generation.csv",
        stops_csv_path="dataset_generator/stopwithZones.csv",
        depot_id=1,     # change to your actual downtown stop ID!
        vehicle_capacity=60,
        save_path="route_designer/generated_routes.json"
    )
