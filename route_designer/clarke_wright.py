import numpy as np

def route_load(route, demand_graph):
    """
    Compute total load on a route based on demand between consecutive stops.
    """
    load = 0
    for a, b in zip(route[:-1], route[1:]):
        load += demand_graph.get(a, {}).get(b, 0)
    return load


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

    # *** Map stop IDs to matrix indices
    id_to_index = {stop_ids[i]: i for i in range(len(stop_ids))}
    depot_idx = id_to_index[depot_id]

    # *** Initialize each stop as its own route: Depot -> stop -> Depot
    routes = {stop: [depot_id, stop, depot_id] for stop in stop_ids if stop != depot_id}

    # *** Compute Savings values for each pair of stops
    savings = []
    for i in stop_ids:
        if i == depot_id: 
            continue
        for j in stop_ids:
            if j == depot_id or j == i: 
                continue

            i_idx = id_to_index[i]
            j_idx = id_to_index[j]

            sij = (
                cost_matrix[depot_idx][i_idx]
                + cost_matrix[depot_idx][j_idx]
                - cost_matrix[i_idx][j_idx]
            )
            savings.append((sij, i, j))

    # *** Sort savings descending
    savings.sort(reverse=True, key=lambda x: x[0])

    # *** Merge routes in order of savings
    for s, i, j in savings:

        # Find routes containing i at the end and j at the start
        route_i = None
        route_j = None

        for r in routes.values():
            # route form: [Depot, ..., i, Depot]
            if r[-2] == i:
                route_i = r
            # route form: [Depot, j, ..., Depot]
            if r[1] == j:
                route_j = r

        # Skip if not found, or same route
        if route_i is None or route_j is None or route_i is route_j:
            continue

     
        # Capacity Constraint Here
        load_i = route_load(route_i, demand_graph)
        load_j = route_load(route_j, demand_graph)

        # If combined load exceeds vehicle capacity, do NOT merge
        if load_i + load_j > vehicle_capacity:
            continue

 
        # Merge only if i is last before depot in route_i
        # and j is first after depot in route_j
        if route_i[-2] == i and route_j[1] == j:
            # New route: remove duplicated depot
            new_route = route_i[:-1] + route_j[1:]

            # Remove old routes
            keys_to_delete = []
            for key, r in routes.items():
                if r == route_i or r == route_j:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del routes[key]

            # Add the new route, key by first non-depot stop
            routes[new_route[1]] = new_route

    # Convert dict routes to list
    final_routes = list(routes.values())

    return final_routes
