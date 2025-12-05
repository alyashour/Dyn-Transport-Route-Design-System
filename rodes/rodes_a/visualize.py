import pandas as pd
import folium
from folium import plugins
import osmnx as ox
import networkx as nx
import random

class RouteVisualizer:
    def __init__(self, routes_csv_path, stops_csv_path, city_name):
        """
        Initialize the route visualizer.
        
        Parameters:
        - routes_csv_path: Path to designed_routes.csv
        - stops_csv_path: Path to STOPS.CSV
        - city_name: City name for map centering
        """
        print(f"Loading routes from {routes_csv_path}...")
        self.routes_df = pd.read_csv(routes_csv_path)
        
        print(f"Loading stops from {stops_csv_path}...")
        self.stops_df = pd.read_csv(stops_csv_path)
        
        print(f"Downloading road network for {city_name}...")
        self.graph = ox.graph_from_place(city_name, network_type='drive')
        
        # Create stop lookup
        self.stops_dict = {}
        for _, row in self.stops_df.iterrows():
            self.stops_dict[str(row['stop_id'])] = {
                'name': row['stop_name'],
                'lat': row['stop_lat'],
                'lon': row['stop_lon']
            }
        
        # Generate colors for routes
        self.colors = self._generate_colors(len(self.routes_df))
        
    def _generate_colors(self, n) -> list[str]:
        """Generate n visually distinct colors."""
        colors = [
            '#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A',
            '#FFD700', '#4B0082', '#FF1493', '#00CED1', '#FF4500',
            '#2E8B57', '#DC143C', '#00FA9A', '#FF6347', '#4682B4'
        ]
        
        # If we need more colors, generate random ones
        while len(colors) < n:
            colors.append('#%06X' % random.randint(0, 0xFFFFFF))
        
        return colors[:n]
    
    def _get_route_geometry(self, stop_ids):
        """
        Get the actual road geometry between stops.
        
        Parameters:
        - stop_ids: List of stop IDs in order
        
        Returns:
        - List of (lat, lon) coordinates following roads
        """
        all_coords = []
        
        for i in range(len(stop_ids) - 1):
            stop1_id = stop_ids[i]
            stop2_id = stop_ids[i + 1]
            
            # Get stop coordinates
            stop1 = self.stops_dict[stop1_id]
            stop2 = self.stops_dict[stop2_id]
            
            # Find nearest nodes in the road network
            node1 = ox.distance.nearest_nodes(
                self.graph, stop1['lon'], stop1['lat']
            )
            node2 = ox.distance.nearest_nodes(
                self.graph, stop2['lon'], stop2['lat']
            )
            
            try:
                # Get shortest path
                route = nx.shortest_path(
                    self.graph, node1, node2, weight='length'
                )
                
                # Extract coordinates from path
                for node in route:
                    node_data = self.graph.nodes[node]
                    all_coords.append((node_data['y'], node_data['x']))
                    
            except nx.NetworkXNoPath:
                # If no path found, draw straight line
                all_coords.append((stop1['lat'], stop1['lon']))
                all_coords.append((stop2['lat'], stop2['lon']))
        
        return all_coords
    
    def create_map(self, output_path='route_map.html', show_all_stops=True):
        """
        Create an interactive map with all routes.
        
        Parameters:
        - output_path: Where to save the HTML map
        - show_all_stops: Whether to show all stops or just those on routes
        """
        print("Creating map...")
        
        # Calculate map center
        center_lat = self.stops_df['stop_lat'].mean()
        center_lon = self.stops_df['stop_lon'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add route lines and stops
        for idx, route_row in self.routes_df.iterrows():
            route_id = route_row['route_id']
            stop_ids = route_row['stop_ids'].split(',')
            color = self.colors[idx] # type: ignore
            
            print(f"  Drawing {route_id} with {len(stop_ids)} stops...")
            
            # Get route geometry following roads
            try:
                route_coords = self._get_route_geometry(stop_ids)
                
                # Draw route line
                folium.PolyLine(
                    route_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=f"{route_id}<br>{route_row['num_stops']} stops<br>{route_row['total_distance_km']} km",
                    tooltip=route_id
                ).add_to(m)
            except Exception as e:
                print(f"    Warning: Could not draw route geometry for {route_id}: {e}")
                # Fall back to straight lines
                coords = [(self.stops_dict[sid]['lat'], self.stops_dict[sid]['lon']) 
                         for sid in stop_ids]
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=f"{route_id}<br>{route_row['num_stops']} stops<br>{route_row['total_distance_km']} km",
                    tooltip=route_id
                ).add_to(m)
            
            # Add stops for this route
            for stop_id in stop_ids:
                stop = self.stops_dict[stop_id]
                
                folium.CircleMarker(
                    location=[stop['lat'], stop['lon']],
                    radius=6,
                    color=color,
                    fill=True,
                    fillColor='white',
                    fillOpacity=0.9,
                    weight=2,
                    popup=f"<b>{stop['name']}</b><br>Stop ID: {stop_id}<br>Route: {route_id}",
                    tooltip=stop['name']
                ).add_to(m)
        
        # Optionally show all stops in gray
        if show_all_stops:
            print("  Adding all stops...")
            for _, stop_row in self.stops_df.iterrows():
                stop_id = str(stop_row['stop_id'])
                
                # Only show stops not already shown
                on_route = False
                for _, route_row in self.routes_df.iterrows():
                    if stop_id in route_row['stop_ids'].split(','):
                        on_route = True
                        break
                
                if not on_route:
                    folium.CircleMarker(
                        location=[stop_row['stop_lat'], stop_row['stop_lon']],
                        radius=4,
                        color='gray',
                        fill=True,
                        fillColor='lightgray',
                        fillOpacity=0.5,
                        weight=1,
                        popup=f"<b>{stop_row['stop_name']}</b><br>Stop ID: {stop_id}<br>(Not on any route)",
                        tooltip=stop_row['stop_name']
                    ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin:0; font-weight: bold;">Routes</p>
        '''
        
        for idx, route_row in self.routes_df.iterrows():
            color = self.colors[idx] # type: ignore
            legend_html += f'''
            <p style="margin:5px 0;">
                <span style="background-color:{color}; width:20px; height:3px; 
                      display:inline-block; margin-right:5px;"></span>
                {route_row['route_id']}
            </p>
            '''
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html)) # type: ignore
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add measure control
        plugins.MeasureControl().add_to(m)
        
        # Save map
        m.save(output_path)
        print(f"\nMap saved to {output_path}")
        print(f"Open it in your browser to view the routes!")
        
        return m
    
    def create_individual_route_maps(self, output_folder='route_maps'):
        """
        Create separate maps for each route.
        
        Parameters:
        - output_folder: Folder to save individual route maps
        """
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Creating individual route maps in {output_folder}/...")
        
        for idx, route_row in self.routes_df.iterrows():
            route_id = route_row['route_id']
            stop_ids = route_row['stop_ids'].split(',')
            color = self.colors[idx] # type: ignore
            
            # Get coordinates for this route
            coords = [(self.stops_dict[sid]['lat'], self.stops_dict[sid]['lon']) 
                     for sid in stop_ids]
            
            # Calculate center
            center_lat = sum(c[0] for c in coords) / len(coords)
            center_lon = sum(c[1] for c in coords) / len(coords)
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                tiles='OpenStreetMap'
            )
            
            # Draw route
            try:
                route_coords = self._get_route_geometry(stop_ids)
                folium.PolyLine(
                    route_coords,
                    color=color,
                    weight=5,
                    opacity=0.8
                ).add_to(m)
            except:
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=5,
                    opacity=0.8
                ).add_to(m)
            
            # Add stops with numbers
            for i, stop_id in enumerate(stop_ids):
                stop = self.stops_dict[stop_id]
                
                folium.CircleMarker(
                    location=[stop['lat'], stop['lon']],
                    radius=8,
                    color=color,
                    fill=True,
                    fillColor='white',
                    fillOpacity=1,
                    weight=3,
                    popup=f"<b>Stop {i+1}: {stop['name']}</b><br>Stop ID: {stop_id}",
                ).add_to(m)
                
                # Add number label
                folium.Marker(
                    location=[stop['lat'], stop['lon']],
                    icon=folium.DivIcon(html=f'''
                        <div style="font-size: 10px; font-weight: bold; 
                                    color: black; text-align: center;">
                            {i+1}
                        </div>
                    ''')
                ).add_to(m)
            
            # Add route info
            info_html = f'''
            <div style="position: fixed; 
                        top: 10px; right: 10px; width: 250px; 
                        background-color: white; z-index:9999; font-size:14px;
                        border:2px solid grey; border-radius: 5px; padding: 10px">
            <h3 style="margin:0 0 10px 0;">{route_id}</h3>
            <p style="margin:5px 0;"><b>Stops:</b> {route_row['num_stops']}</p>
            <p style="margin:5px 0;"><b>Distance:</b> {route_row['total_distance_km']} km</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(info_html)) # type: ignore
            
            # Save
            output_path = os.path.join(output_folder, f'{route_id}.html')
            m.save(output_path)
            print(f"  Saved {output_path}")
        
        print(f"\nAll individual maps saved!")


# Example usage
if __name__ == "__main__":
    # Create visualizer
    viz = RouteVisualizer(
        routes_csv_path='designed_routes.csv',
        stops_csv_path='/data/stops.csv',
        city_name='London, Ontario, Canada'
    )
    
    # Create main map with all routes
    viz.create_map(
        output_path='all_routes_map.html',
        show_all_stops=True  # Show stops not on any route in gray
    )
    
    # Create individual maps for each route
    viz.create_individual_route_maps(output_folder='individual_route_maps')
    
    print("\n✓ Visualization complete!")
    print("  - Open 'all_routes_map.html' to see all routes together")
    print("  - Check 'individual_route_maps/' folder for individual route maps")