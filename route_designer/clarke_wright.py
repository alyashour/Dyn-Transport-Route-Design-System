import numpy as np

def clarke_wright_routes(demand_graph, cost_matrix, stop_ids, depot_id, vehicle_capacity=60):
    """
    Clarke–Wright Savings heuristic for generating bus routes.

    Inputs:
        demand_graph : dict-of-dicts
            demand_graph[o][d] = number of trips from o -> d
        cost_matrix : numpy array
            cost_matrix[i][j] = distance between stop_i and stop_j
        stop_ids : list of stop IDs corresponding to cost_matrix indices
        depot_id : ID of the downtown depot
        vehicle_capacity : max load per bus route

    Output:
        routes : list of lists
            Each route is a sequence of stop IDs, starting/ending at depot
    """

    # Map stop IDs to matrix indices
    id_to_index = {stop_ids[i]: i for i in range(len(stop_ids))}
    depot_idx = id_to_index[depot_id]

    # Step 1: Initialize each stop as its own route: Depot -> stop -> Depot
    routes = {stop: [depot_id, stop, depot_id] for stop in stop_ids if stop != depot_id}

    # Step 2: Compute Savings values for each pair of stops
    savings = []

    for i in stop_ids:
        if i == depot_id: continue
        for j in stop_ids:
            if j == depot_id or j == i: continue

            i_idx = id_to_index[i]
            j_idx = id_to_index[j]

            # Clarke-Wright formula
            sij = (
                cost_matrix[depot_idx][i_idx]
                + cost_matrix[depot_idx][j_idx]
                - cost_matrix[i_idx][j_idx]
            )

            savings.append((sij, i, j))

    # Sort by decreasing savings
    savings.sort(reverse=True, key=lambda x: x[0])

    # Step 3: Try to merge routes in order of savings
    for s, i, j in savings:

        # Find routes containing i and j
        route_i = None
        route_j = None

        for r in routes.values():
            if r[1] == i: route_i = r
            if r[-2] == j: route_j = r

        # If either not found, or same route, skip
        if route_i is None or route_j is None or route_i is route_j:
            continue

        # Check capacity (optional for now)
        # For bus ridership, sum demand on the route
        # For now, we allow merges (you can refine later)

        # Merge condition: i at end of route_i, j at start of route_j
        if route_i[-2] == i and route_j[1] == j:
            # Merge them: remove duplicated depot from middle
            new_route = route_i[:-1] + route_j[1:]

            # Remove old routes
            keys_to_delete = []
            for key, r in routes.items():
                if r == route_i or r == route_j:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del routes[key]

            # Add new route, key by starting stop
            routes[new_route[1]] = new_route

    # Convert dict routes → list
    final_routes = list(routes.values())

    return final_routes
