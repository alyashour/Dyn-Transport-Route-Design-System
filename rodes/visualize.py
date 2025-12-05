<<<<<<< HEAD
<<<<<<< HEAD
import sys
=======
>>>>>>> 0902e9d (added google ortools)
=======
import sys
>>>>>>> 1c4ebaa (added gtfs to route and arg parsing in visualize)
import pandas as pd
import osmnx as ox
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 1c4ebaa (added gtfs to route and arg parsing in visualize)
args = sys.argv
if len(args) < 2:
    print('Usage `visualize.py <routes.csv>`')
    sys.exit(1)
else:
    ROUTES_CSV = args[1]

# ROUTES_CSV = "rodes_a/designed_routes/google_ortools-2025-11-18.csv"
<<<<<<< HEAD
=======
ROUTES_CSV = "rodes_a/designed_routes/google_ortools-2025-11-18.csv"
>>>>>>> 0902e9d (added google ortools)
=======
>>>>>>> 1c4ebaa (added gtfs to route and arg parsing in visualize)
STOPS_CSV = "../data/stops.csv"

routes_df = pd.read_csv(ROUTES_CSV)
stops_df = pd.read_csv(STOPS_CSV)

# Build stop lookup
stop_lookup = {row["stop_id"]: (row["stop_lat"], row["stop_lon"]) for _, row in stops_df.iterrows()}

# -----------------------------------------
# Determine bounding box from all routes
# -----------------------------------------
all_coords = []
for _, route in routes_df.iterrows():
    stop_ids = [s.strip() for s in route["stops"].split("→")]
    for sid in stop_ids:
        all_coords.append(stop_lookup[sid])

lats = [c[0] for c in all_coords]
lons = [c[1] for c in all_coords]

padding = 0.002
north = max(lats) + padding
south = min(lats) - padding
east = max(lons) + padding
west = min(lons) - padding

# -----------------------------------------
# Download graph
# -----------------------------------------
G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")

# -----------------------------------------
# Prepare Plotly figure
# -----------------------------------------
fig = go.Figure()

color_cycle = cycle(px.colors.qualitative.Dark24)  # cycle through 24 distinct colors

for _, route in routes_df.iterrows():
    stop_ids = [s.strip() for s in route["stops"].split("→")]
    coords = [stop_lookup[s] for s in stop_ids]

    # Nearest nodes
    nodes = [ox.nearest_nodes(G, X=lon, Y=lat) for lat, lon in coords]

    # Build route path
    route_nodes = []
    for i in range(len(nodes) - 1):
        path = ox.shortest_path(G, nodes[i], nodes[i + 1], weight="length")
        route_nodes.extend(path)

    xs = [G.nodes[n]["x"] for n in route_nodes]
    ys = [G.nodes[n]["y"] for n in route_nodes]

    color = next(color_cycle)

    # Route line
    fig.add_trace(go.Scattermapbox(
        lon=xs, lat=ys,
        mode="lines",
        line=dict(width=4, color=color),
        name=f"Route {route['route_id']}",
        hoverinfo="name"
    ))

    # Stops
    stop_x = [lon for lat, lon in coords]
    stop_y = [lat for lat, lon in coords]
    fig.add_trace(go.Scattermapbox(
        lon=stop_x, lat=stop_y,
        mode="markers",
        marker=dict(size=8, color=color),
        name=f"Stops {route['route_id']}",
        hoverinfo="name"
    ))

# -----------------------------------------
# Configure Mapbox
# -----------------------------------------
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=13,
    mapbox_center={"lat": (north+south)/2, "lon": (east+west)/2},
    margin={"r":0,"t":0,"l":0,"b":0},
    legend=dict(itemsizing='constant')
)

fig.show()
