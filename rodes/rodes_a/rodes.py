import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import osmnx as ox
import networkx as nx
from collections import defaultdict

@dataclass
class Stop:
    id: str
    lat: float
    lon: float
    name: str

@dataclass
class Route:
    stops: List[str]
    total_distance: float

class TransitRouteDesigner:
    def __init__(self, stops_csv_path: str, trips_npy_path: str, city_name: str):
        """
        Initialize the route designer.
        
        Parameters:
        - stops_csv_path: Path to STOPS.CSV file
        - trips_npy_path: Path to trips .npy file (2000x2000 OD matrix)
        - city_name: City name for OSMnx (e.g., "London, Ontario, Canada")
        """
        print(f"Loading stops from {stops_csv_path}...")
        self.stops_df = pd.read_csv(stops_csv_path)
        self.stops = self._load_stops()
        
        print(f"Loading trips matrix from {trips_npy_path}...")
        self.trips_matrix = np.load(trips_npy_path)
        
        print(f"Downloading road network for {city_name} from OpenStreetMap...")
        self.road_graph = ox.graph_from_place(city_name, network_type='drive')
        
        print("Calculating distances between stops...")
        self.distances = self._calculate_distances_from_osm()
        
        self.routes = []
        print("Initialization complete!")
        
    def _load_stops(self):
        """Load stops from GTFS format CSV."""
        stops = {}
        for _, row in self.stops_df.iterrows():
            stop_id = str(row['stop_id'])
            stops[stop_id] = Stop(
                id=stop_id,
                lat=row['stop_lat'],
                lon=row['stop_lon'],
                name=row.get('stop_name', stop_id)
            )
        return stops
    
    def _calculate_distances_from_osm(self):
        """Calculate shortest path distances between all stops using OSM road network."""
        distances = defaultdict(lambda: defaultdict(lambda: float('inf')))
        stop_ids = list(self.stops.keys())
        
        # Find nearest network nodes for each stop
        stop_nodes = {}
        for stop_id, stop in self.stops.items():
            try:
                nearest_node = ox.distance.nearest_nodes(
                    self.road_graph, 
                    stop.lon, 
                    stop.lat
                )
                stop_nodes[stop_id] = nearest_node
            except Exception as e:
                print(f"Warning: Could not map stop {stop_id} to road network: {e}")
        
        # Calculate pairwise distances
        total_pairs = len(stop_ids) * (len(stop_ids) - 1) // 2
        calculated = 0
        
        for i, stop_i in enumerate(stop_ids):
            if stop_i not in stop_nodes:
                continue
                
            # Self-distance
            distances[stop_i][stop_i] = 0
            
            for stop_j in stop_ids[i+1:]:
                if stop_j not in stop_nodes:
                    continue
                
                try:
                    # Calculate shortest path distance in meters
                    length = nx.shortest_path_length(
                        self.road_graph,
                        stop_nodes[stop_i],
                        stop_nodes[stop_j],
                        weight='length'
                    )
                    # Convert to kilometers
                    length_km = length / 1000.0
                    
                    distances[stop_i][stop_j] = length_km
                    distances[stop_j][stop_i] = length_km
                    
                except nx.NetworkXNoPath:
                    # No path exists, use large distance
                    distances[stop_i][stop_j] = 999999
                    distances[stop_j][stop_i] = 999999
                
                calculated += 1
                if calculated % 100 == 0:
                    print(f"  Calculated {calculated}/{total_pairs} distances...")
        
        print(f"  Distance calculation complete!")
        return distances
    
    def _get_distance(self, stop1, stop2):
        """Get distance between two stops."""
        return self.distances[stop1][stop2]
    
    def _get_demand(self, stop_i, stop_j):
        """Get demand (trips) between two stops from the trips matrix."""
        try:
            # Convert stop IDs to indices
            idx_i = list(self.stops.keys()).index(stop_i)
            idx_j = list(self.stops.keys()).index(stop_j)
            
            # Get bidirectional demand
            demand = self.trips_matrix[idx_i, idx_j] + self.trips_matrix[idx_j, idx_i]
            
            # Normalize
            max_demand = self.trips_matrix.max()
            return demand / max_demand if max_demand > 0 else 0
        except (ValueError, IndexError):
            return 0
    
    def _calculate_savings(self, depot=None):
        """
        Calculate Clark-Wright savings for all pairs of stops.
        Savings(i,j) = distance(depot,i) + distance(depot,j) - distance(i,j)
        """
        stop_ids = list(self.stops.keys())
        
        # Use first stop as depot if not specified
        if depot is None:
            depot = stop_ids[0]
        
        print(f"Calculating Clark-Wright savings (using {depot} as depot)...")
        savings = []
        
        for i in range(len(stop_ids)):
            for j in range(i + 1, len(stop_ids)):
                stop_i = stop_ids[i]
                stop_j = stop_ids[j]
                
                if stop_i == depot or stop_j == depot:
                    continue
                
                # Calculate savings
                d_depot_i = self._get_distance(depot, stop_i)
                d_depot_j = self._get_distance(depot, stop_j)
                d_i_j = self._get_distance(stop_i, stop_j)
                
                # Skip if any distance is invalid
                if d_depot_i == float('inf') or d_depot_j == float('inf') or d_i_j == float('inf'):
                    continue
                
                s = d_depot_i + d_depot_j - d_i_j
                
                # Weight by demand between stops
                demand = self._get_demand(stop_i, stop_j)
                weighted_savings = s * (1 + 0.5 * demand)  # 50% demand weight
                
                savings.append((weighted_savings, stop_i, stop_j))
        
        # Sort by savings (descending)
        savings.sort(reverse=True)
        print(f"  Calculated {len(savings)} savings values")
        return savings
    
    def design_routes(self, max_route_length=15, max_distance=30, depot=None):
        """
        Design routes using Clark-Wright Savings algorithm.
        
        Parameters:
        - max_route_length: Maximum number of stops per route (default: 15)
        - max_distance: Maximum total distance for a route in km (default: 30)
        - depot: Optional depot stop (uses first stop if None)
        """
        stop_ids = list(self.stops.keys())
        
        if depot is None:
            depot = stop_ids[0]
        
        print(f"\nDesigning routes with Clark-Wright algorithm...")
        print(f"  Max stops per route: {max_route_length}")
        print(f"  Max distance per route: {max_distance} km")
        
        # Calculate savings
        savings = self._calculate_savings(depot)
        
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
        
        print(f"  Completed {merges} route merges")
        
        # Convert to Route objects
        self.routes = []
        for route_stops in routes.values():
            distance = self._calculate_route_distance(route_stops)
            self.routes.append(Route(route_stops, distance))
        
        print(f"\nCreated {len(self.routes)} routes")
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
            dist = self._get_distance(stops[i], stops[i + 1])
            if dist == float('inf'):
                return float('inf')
            total += dist
        return total
    
    def calculate_metrics(self, service_frequency_minutes=15, service_hours=16, 
                         bus_capacity=50, days_in_analysis=None):
        """
        Calculate average wait time and route utilization.
        
        Parameters:
        - service_frequency_minutes: How often buses run (minutes between buses)
        - service_hours: Hours of service per day
        - bus_capacity: Seats per bus
        - days_in_analysis: Number of days in trip data (auto-detect if None)
        """
        print("\nCalculating performance metrics...")
        
        # Average wait time (simple model: half the headway)
        avg_wait_time = service_frequency_minutes / 2
        
        # Calculate route utilization
        route_utilizations = []
        
        for route_idx, route in enumerate(self.routes):
            # Get all trips that could use this route
            route_stop_ids = [s for s in route.stops if s in self.stops]
            
            # Get indices for these stops
            stop_indices = []
            for stop_id in route_stop_ids:
                try:
                    idx = list(self.stops.keys()).index(stop_id)
                    stop_indices.append(idx)
                except ValueError:
                    continue
            
            if len(stop_indices) < 2:
                route_utilizations.append(0)
                continue
            
            # Sum all trips between stops on this route
            total_trips = 0
            for i in stop_indices:
                for j in stop_indices:
                    if i != j:
                        total_trips += self.trips_matrix[i, j]
            
            # Calculate daily demand (assuming trips_matrix is total over some period)
            # If you know the number of days, pass it in
            if days_in_analysis:
                daily_demand = total_trips / days_in_analysis
            else:
                # Assume it's already daily or use as-is
                daily_demand = total_trips
            
            # Calculate capacity (buses per day * seats per bus)
            buses_per_day = (service_hours * 60) / service_frequency_minutes
            capacity = buses_per_day * bus_capacity
            
            # Utilization percentage
            utilization = (daily_demand / capacity * 100) if capacity > 0 else 0
            route_utilizations.append(min(utilization, 100))  # Cap at 100%
        
        avg_utilization = np.mean(route_utilizations) if route_utilizations else 0
        
        metrics = {
            'avg_wait_time_minutes': avg_wait_time,
            'avg_route_utilization_pct': avg_utilization,
            'route_utilizations': route_utilizations
        }
        
        print(f"\nMetrics Summary:")
        print(f"  Average Wait Time: {metrics['avg_wait_time_minutes']:.2f} minutes")
        print(f"  Average Route Utilization: {metrics['avg_route_utilization_pct']:.2f}%")
        
        return metrics
    
    def export_routes(self, output_path: Optional[str] = None):
        """Export routes to DataFrame and optionally save to CSV."""
        route_data = []
        for idx, route in enumerate(self.routes):
            # Get stop names
            stop_names = []
            for stop_id in route.stops:
                if stop_id in self.stops:
                    stop_names.append(self.stops[stop_id].name)
                else:
                    stop_names.append(stop_id)
            
            route_data.append({
                'route_id': f'Route_{idx + 1}',
                'stops': ' → '.join(route.stops),
                'stop_names': ' → '.join(stop_names),
                'num_stops': len(route.stops),
                'total_distance_km': round(route.total_distance, 2)
            })
        
        df = pd.DataFrame(route_data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nRoutes exported to {output_path}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize with your data
    designer = TransitRouteDesigner(
        stops_csv_path='../../data/stops.csv',
        trips_npy_path='../../data/mlr_output_2021_11_18.npy',
        city_name='London, Ontario, Canada'
    )
    
    # Design routes
    routes = designer.design_routes(
        max_route_length=15,    # Max stops per route
        max_distance=30         # Max km per route
    )
    
    # Calculate metrics
    metrics = designer.calculate_metrics(
        service_frequency_minutes=15,  # Bus every 15 minutes
        service_hours=16,               # 16 hours of service per day
        bus_capacity=50,                # 50 seats per bus
        days_in_analysis=30             # If your trips matrix is over 30 days
    )
    
    # Export results
    routes_df = designer.export_routes('designed_routes.csv')
    print("\n" + "="*60)
    print("DESIGNED ROUTES:")
    print("="*60)
    print(routes_df.to_string(index=False))