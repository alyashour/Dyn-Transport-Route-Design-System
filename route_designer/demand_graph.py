import pandas as pd
from collections import defaultdict

def demand_to_graph(demand_matrix, stop_ids):
    graph = {}
    n = len(stop_ids)

    for i in range(n):
        origin = stop_ids[i]
        graph[origin] = {}
        for j in range(n):
            dest = stop_ids[j]
            graph[origin][dest] = demand_matrix[i, j]

    return graph

