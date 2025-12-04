import json
import pandas as pd
import folium

# Load stops
stops = pd.read_csv("dataset_generator/stopwithZones.csv")

# Remove duplicates
stops = stops.drop_duplicates(subset='stop_id')

# Create index
stop_locs = stops.set_index("stop_id")[["stop_lat","stop_lon"]]

# Load routes
with open("route_designer/generated_routes.json","r") as f:
    routes = json.load(f)

m = folium.Map(location=[42.9833, -81.25], zoom_start=12)

colors = ["red","blue","green","purple","orange","darkred",
          "darkblue","darkgreen","cadetblue","black"]

for r_index, route in enumerate(routes):

    coords = []

    for stop in route:
        try:
            lat = float(stop_locs.loc[stop, "stop_lat"])
            lon = float(stop_locs.loc[stop, "stop_lon"])
            coords.append([lat, lon])
        except KeyError:
            print(f"Warning: Stop {stop} not found in stop_locs")
            continue

    if len(coords) > 1:
        folium.PolyLine(coords,
                color=colors[r_index % len(colors)],
                weight=3,
                opacity=0.8).add_to(m)

        for lat, lon in coords:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color="black",
                fill=True
            ).add_to(m)

m.save("routes_map.html")
print("Map saved to route_designer/routes_map.html")
