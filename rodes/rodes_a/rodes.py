import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import heapq
from collections import defaultdict
import osmnx as ox
import networkx as nx

class Stop:
    def __init__(self, id: str, name: str, lat: float, lon: float):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon

class Route:
    def __init__(self, stops: List[str], total_distance: float):
        self.stops = stops
        self.total_distance = total_distance

class TransitRouteDesigner:
    def __init__(self, stops_csv_path, trips_csv_path, city_name):
        """
        Initialize the route designer.
        
        Parameters:
        - stops_csv_path: Path to STOPS.CSV file
        - trips_csv_path: Path to TRIPS.CSV file  
        - city_name: City name for OSMnx (e.g., "London, Ontario, Canada")
        """
        print(f"Loading stops from {stops_csv_path}...")
        self.stops_df = pd.read_csv(stops_csv_path)
        self.stops = self._load_stops()
        
        print(f"Loading trip data from {trips_csv_path}...")
        self.od_data = pd.read_csv(trips_csv_path)
        self._process_trip_data()
        
        print(f"Downloading road network for {city_name} from OpenStreetMap...")
        self.graph = ox.graph_from_place(city_name, network_type='drive')
        print(f"Road network loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        
        self.distances = {}
        self.routes = []
        
    def _load_stops(self):
        """Load stops into dictionary."""
        stops = {}
        for _, row in self.stops_df.iterrows():
            stops[str(row['stop_id'])] = Stop(
                id=str(row['stop_id']),
                name=row['stop_name'],
                lat=row['stop_lat'],
                lon=row['stop_lon']
            )
        return stops
    
    def _process_trip_data(self):
        """Process trip data to add date column and normalize."""
        # Create date column from Day, Month, Year
        self.od_data['date'] = pd.to_datetime(
            self.od_data[['Year', 'Month', 'Day']].rename(
                columns={'Year': 'year', 'Month': 'month', 'Day': 'day'}
            )
        )
        
        # Normalize column names
        self.od_data.rename(columns={
            'Origin ID': 'origin',
            'Destination ID': 'destination'
        }, inplace=True)
        
        # Convert to string IDs
        self.od_data['origin'] = self.od_data['origin'].astype(str)
        self.od_data['destination'] = self.od_data['destination'].astype(str)
        
        # Count trips per OD pair
        self.od_data['trips'] = 1
        self.od_data = self.od_data.groupby(
            ['origin', 'destination', 'date'], as_index=False
        ).agg({'trips': 'sum'})
        
        print(f"Processed {len(self.od_data)} unique OD pairs")
        print(f"Total trips: {self.od_data['trips'].sum()}")
    
    def precompute_distances(self, stop_ids=None):
        """
        Precompute shortest path distances between stops using road network.
        This can take a while for many stops - only compute what's needed.
        """
        if stop_ids is None:
            stop_ids = list(self.stops.keys())
        
        print(f"Precomputing distances for {len(stop_ids)} stops...")
        
        # Find nearest network nodes for each stop
        stop_nodes = {}
        for stop_id in stop_ids:
            stop = self.stops[stop_id]
            nearest_node = ox.distance.nearest_nodes(
                self.graph, stop.lon, stop.lat
            )
            stop_nodes[stop_id] = nearest_node
        
        # Compute shortest paths
        for i, stop1 in enumerate(stop_ids):
            if i % 10 == 0:
                print(f"  Processing stop {i+1}/{len(stop_ids)}...")
            
            node1 = stop_nodes[stop1]
            
            # Use Dijkstra from this node to all others
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, node1, weight='length'
            )
            
            for stop2 in stop_ids:
                node2 = stop_nodes[stop2]
                
                # Distance in meters, convert to km
                if node2 in lengths:
                    dist_km = lengths[node2] / 1000.0
                else:
                    dist_km = float('inf')
                
                if stop1 not in self.distances:
                    self.distances[stop1] = {}
                self.distances[stop1][stop2] = dist_km
        
        print("Distance computation complete!")
    
    def _get_distance(self, stop1, stop2):
        """Get distance between two stops."""
        if stop1 in self.distances and stop2 in self.distances[stop1]:
            return self.distances[stop1][stop2]
        return float('inf')
    
    def _calculate_savings(self, depot=None):
        """
        Calculate Clark-Wright savings for all pairs of stops.
        Savings(i,j) = distance(depot,i) + distance(depot,j) - distance(i,j)
        """
        stop_ids = list(self.stops.keys())
        
        # Use first stop as depot if not specified
        if depot is None:
            depot = stop_ids[0]
        
        savings = []
        
        for i in range(len(stop_ids)):
            for j in range(i + 1, len(stop_ids)):
                stop_i = stop_ids[i]
                stop_j = stop_ids[j]
                
                if stop_i == depot or stop_j == depot:
                    continue
                
                # Calculate savings
                s = (self._get_distance(depot, stop_i) + 
                     self._get_distance(depot, stop_j) - 
                     self._get_distance(stop_i, stop_j))
                
                # Weight by demand between stops
                demand = self._get_demand(stop_i, stop_j)
                weighted_savings = s * (1 + 0.5 * demand)  # 50% demand weight
                
                savings.append((weighted_savings, stop_i, stop_j))
        
        # Sort by savings (descending)
        savings.sort(reverse=True)
        return savings
    
    def _get_demand(self, stop_i, stop_j):
        """Get demand (trips) between two stops."""
        demand = self.od_data[
            ((self.od_data['origin'] == stop_i) & (self.od_data['destination'] == stop_j)) |
            ((self.od_data['origin'] == stop_j) & (self.od_data['destination'] == stop_i))
        ]['trips'].sum()
        
        # Normalize to avoid overwhelming the savings calculation
        max_demand = self.od_data['trips'].max()
        return demand / max_demand if max_demand > 0 else 0
    
    def design_routes(self, max_route_length=10, max_distance=50, depot=None):
        """
        Design routes using Clark-Wright Savings algorithm.
        
        Parameters:
        - max_route_length: Maximum number of stops per route
        - max_distance: Maximum total distance for a route (km)
        - depot: Optional depot stop (uses first stop if None)
        """
        stop_ids = list(self.stops.keys())
        
        if depot is None:
            depot = stop_ids[0]
        
        print(f"Designing routes with depot at {self.stops[depot].name}...")
        
        # Calculate savings
        savings = self._calculate_savings(depot)
        print(f"Calculated {len(savings)} potential connections")
        
        # Initialize: each stop is its own route from depot
        routes = {stop: [depot, stop, depot] for stop in stop_ids if stop != depot}
        stop_to_route = {stop: stop for stop in stop_ids if stop != depot}
        
        # Process savings
        merges = 0
        for saving, stop_i, stop_j in savings:
            if stop_i not in stop_to_route or stop_j not in stop_to_route:
                continue
            
            route_i = stop_to_route[stop_i]
            route_j = stop_to_route[stop_j]
            
            if route_i == route_j:
                continue  # Already in same route
            
            route_i_stops = routes[route_i]
            route_j_stops = routes[route_j]
            
            # Check if stops are at route ends
            i_is_end = (route_i_stops[1] == stop_i or route_i_stops[-2] == stop_i)
            j_is_end = (route_j_stops[1] == stop_j or route_j_stops[-2] == stop_j)
            
            if not (i_is_end and j_is_end):
                continue
            
            # Try to merge routes
            merged = self._try_merge_routes(route_i_stops, route_j_stops, stop_i, stop_j, 
                                           max_route_length, max_distance)
            
            if merged:
                # Update routes
                routes[route_i] = merged
                del routes[route_j]
                
                # Update stop-to-route mapping
                for stop in merged[1:-1]:  # Exclude depot at start and end
                    stop_to_route[stop] = route_i
                
                merges += 1
        
        print(f"Completed {merges} route merges")
        
        # Convert to Route objects
        self.routes = []
        for route_stops in routes.values():
            distance = self._calculate_route_distance(route_stops)
            self.routes.append(Route(stops=route_stops, total_distance=distance))
        
        print(f"Created {len(self.routes)} routes")
        return self.routes
    
    def _try_merge_routes(self, route1, route2, stop_i, stop_j, max_length, max_distance):
        """Try to merge two routes."""
        # Remove depot from ends
        r1 = route1[1:-1]
        r2 = route2[1:-1]
        
        # Check length constraint
        if len(r1) + len(r2) > max_length:
            return None
        
        # Determine merge order
        merged = None
        
        if r1[0] == stop_i and r2[0] == stop_j:
            merged = list(reversed(r1)) + r2
        elif r1[0] == stop_i and r2[-1] == stop_j:
            merged = list(reversed(r1)) + list(reversed(r2))
        elif r1[-1] == stop_i and r2[0] == stop_j:
            merged = r1 + r2
        elif r1[-1] == stop_i and r2[-1] == stop_j:
            merged = r1 + list(reversed(r2))
        
        if merged:
            # Add depot at start and end
            merged = [route1[0]] + merged + [route1[0]]
            
            # Check distance constraint
            distance = self._calculate_route_distance(merged)
            if distance <= max_distance:
                return merged
        
        return None
    
    def _calculate_route_distance(self, stops):
        """Calculate total distance for a route."""
        total = 0
        for i in range(len(stops) - 1):
            total += self._get_distance(stops[i], stops[i + 1])
        return total
    
    def calculate_metrics(self, service_frequency_minutes=15, service_hours=16, bus_capacity=50):
        """
        Calculate average wait time and route utilization.
        
        Parameters:
        - service_frequency_minutes: How often buses run (minutes between buses)
        - service_hours: Hours of service per day
        - bus_capacity: Number of seats per bus
        """
        # Average wait time (simple model: half the headway)
        avg_wait_time = service_frequency_minutes / 2
        
        # Calculate route utilization
        route_utilizations = []
        
        for route_idx, route in enumerate(self.routes):
            # Get all trips that could use this route
            route_stops = set(route.stops)
            
            # Find trips where both origin and destination are on this route
            relevant_trips = self.od_data[
                self.od_data['origin'].isin(route_stops) & 
                self.od_data['destination'].isin(route_stops)
            ]
            
            # Calculate daily demand
            if len(relevant_trips) > 0:
                daily_demand = relevant_trips.groupby('date')['trips'].sum().mean()
            else:
                daily_demand = 0
            
            # Calculate capacity (buses per day * seats per bus)
            buses_per_day = (service_hours * 60) / service_frequency_minutes
            capacity = buses_per_day * bus_capacity
            
            # Utilization percentage
            utilization = (daily_demand / capacity * 100) if capacity > 0 else 0
            route_utilizations.append(utilization)
        
        avg_utilization = np.mean(route_utilizations) if route_utilizations else 0
        
        return {
            'avg_wait_time_minutes': avg_wait_time,
            'avg_route_utilization_pct': avg_utilization,
            'route_utilizations': route_utilizations
        }
    
    def export_routes(self):
        """Export routes to DataFrame with stop names."""
        route_data = []
        for idx, route in enumerate(self.routes):
            stop_names = [self.stops[sid].name for sid in route.stops]
            
            route_data.append({
                'route_id': f'Route_{idx + 1}',
                'stops': ' → '.join(stop_names),
                'stop_ids': ','.join(route.stops),
                'num_stops': len(route.stops),
                'total_distance_km': round(route.total_distance, 2)
            })
        
        return pd.DataFrame(route_data)


# Example usage
if __name__ == "__main__":
    # Initialize with your CSV files
    designer = TransitRouteDesigner(
        stops_csv_path='../../data/stops.csv',
        trips_csv_path='../../data/trips.csv',
        city_name='London, Ontario, Canada'
    )
    
    # Precompute distances between all stops (this may take a while)
    designer.precompute_distances()
    
    # Design routes
    routes = designer.design_routes(
        max_route_length=12,  # Max stops per route
        max_distance=25,      # Max route length in km
    )
    
    # Calculate metrics
    metrics = designer.calculate_metrics(
        service_frequency_minutes=15,  # Bus every 15 minutes
        service_hours=16,              # 16 hours of service
        bus_capacity=50                # 50 seats per bus
    )
    
    # Display results
    print("\n" + "="*80)
    print("DESIGNED ROUTES:")
    print("="*80)
    routes_df = designer.export_routes()
    print(routes_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS:")
    print("="*80)
    print(f"Average Wait Time: {metrics['avg_wait_time_minutes']:.2f} minutes")
    print(f"Average Route Utilization: {metrics['avg_route_utilization_pct']:.2f}%")
    
    print("\n" + "="*80)
    print("INDIVIDUAL ROUTE UTILIZATION:")
    print("="*80)
    for idx, util in enumerate(metrics['route_utilizations']):
        print(f"Route {idx+1}: {util:.2f}%")
    
    # Save results
    routes_df.to_csv('designed_routes.csv', index=False)
    print("\nRoutes saved to 'designed_routes.csv'")