import pandas as pd
from collections import defaultdict

def load_ripr_a_grid(csv_path):
    """
    Loads RIPR-A output grid and converts it into a demand graph 
    compatible with RODES.
    """

    df = pd.read_csv(csv_path)

    origins = df.iloc[:, 0].tolist()
    destinations = df.columns[1:].tolist()

    demand_graph = defaultdict(lambda: defaultdict(int))

    for i, origin in enumerate(origins):
        for j, dest in enumerate(destinations):
            trips = df.iloc[i, j+1]
            if trips > 0:
                demand_graph[origin][dest] = trips

    return demand_graph, origins
