import pandas as pd
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import osmnx as ox
import networkx as nx
from scipy.spatial.distance import cdist
import pickle
import os
from tqdm import tqdm

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
    def __init__(self, stops_csv_path: str, trips_npy_path: str, 
                 city_name: str, use_euclidean=False, cache_file=None):
        """
        Initialize the route designer.
        
        Parameters:
        - stops_csv_path: Path to STOPS.CSV file
        - trips_npy_path: Path to trips .npy file (2000x2000 OD matrix)
        - city_name: City name for OSMnx (e.g., "London, Ontario, Canada")
        - use_euclidean: Use fast Euclidean distance instead of road network (default: False)
        - cache_file: Path to cache distances (loads if exists, saves if not)
        """
        print(f"Loading stops from {stops_csv_path}...")
        self.stops_df = pd.read_csv(stops_csv_path)
        self.stops = self._load_stops()
        
        print(f"Loading trips matrix from {trips_npy_path}...")
        self.trips_matrix = np.load(trips_npy_path)
        
        # Check cache first
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached distances from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.distances = pickle.load(f)
            self.road_graph = None  # Don't need graph if using cache
        elif use_euclidean:
            print("Using fast Euclidean distance approximation...")
            self.distances = self._calculate_euclidean_distances()
            self.road_graph = None
            if cache_file:
                self._save_cache(cache_file)
        else:
            print(f"Downloading road network for {city_name}...")
            self.road_graph = ox.graph_from_place(city_name, network_type='drive')
            print("Calculating distances (this will take a while for 2000 stops)...")
            print("Consider using use_euclidean=True for 100x speedup!")
            self.distances = self._calculate_distances_from_osm_optimized()
            if cache_file:
                self._save_cache(cache_file)
        
        self.routes = []
        print("Initialization complete!")
    
    def _save_cache(self, cache_file):
        """Save distance matrix to cache."""
        print(f"Saving distances to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.distances, f)
    
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
    
    def _calculate_euclidean_distances(self):
        """Fast Euclidean distance calculation using vectorization."""
        stop_ids = list(self.stops.keys())
        n = len(stop_ids)
        
        # Create coordinate matrix
        coords = np.array([[self.stops[sid].lat, self.stops[sid].lon] for sid in stop_ids])
        
        # Calculate pairwise Euclidean distances (in degrees)
        dist_matrix = cdist(coords, coords, metric='euclidean')
        
        # Convert to approximate km (rough approximation: 1 degree ≈ 111 km)
        # More accurate: account for latitude
        avg_lat = np.mean(coords[:, 0])
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(avg_lat))
        
        # Weighted distance
        lat_diff = coords[:, 0:1] - coords[:, 0:1].T
        lon_diff = coords[:, 1:2] - coords[:, 1:2].T
        dist_km = np.sqrt((lat_diff * km_per_deg_lat)**2 + (lon_diff * km_per_deg_lon)**2)
        
        # Convert to dictionary format
        distances = {}
        for i, sid_i in enumerate(stop_ids):
            distances[sid_i] = {}
            for j, sid_j in enumerate(stop_ids):
                distances[sid_i][sid_j] = dist_km[i, j]
        
        return distances
    
    def _calculate_distances_from_osm_optimized(self):
        """Optimized OSM distance calculation - only for nearby stops."""
        distances = {}
        stop_ids = list(self.stops.keys())
        
        # First, calculate Euclidean distances
        print("  Step 1/3: Calculating Euclidean distances...")
        euclidean_dist = self._calculate_euclidean_distances()
        
        # Map stops to nearest nodes
        print("  Step 2/3: Mapping stops to road network...")
        stop_nodes = {}
        for stop_id, stop in self.stops.items():
            try:
                nearest_node = ox.distance.nearest_nodes(
                    self.road_graph, stop.lon, stop.lat # type: ignore
                ) # type: ignore
                stop_nodes[stop_id] = nearest_node
            except Exception as e:
                print(f"Warning: Could not map stop {stop_id}: {e}")
        
        # Only calculate exact distances for nearby stops (within threshold)
        print("  Step 3/3: Calculating road distances for nearby pairs...")
        threshold_km = 5.0  # Only calculate exact distance if within 5km Euclidean
        
        for i, sid_i in enumerate(stop_ids):
            if sid_i not in stop_nodes:
                distances[sid_i] = {sid_j: euclidean_dist[sid_i][sid_j] for sid_j in stop_ids}
                continue
            
            distances[sid_i] = {}
            
            for sid_j in stop_ids:
                if sid_j not in stop_nodes:
                    distances[sid_i][sid_j] = euclidean_dist[sid_i][sid_j]
                    continue
                
                euc_dist = euclidean_dist[sid_i][sid_j]
                
                # Only calculate road distance if close enough
                if euc_dist <= threshold_km:
                    try:
                        road_dist = nx.shortest_path_length( # type: ignore
                            self.road_graph, # type: ignore
                            stop_nodes[sid_i],
                            stop_nodes[sid_j],
                            weight='length'
                        ) / 1000.0
                        distances[sid_i][sid_j] = road_dist
                    except nx.NetworkXNoPath:
                        distances[sid_i][sid_j] = euc_dist * 1.4  # Assume 40% detour
                else:
                    # Use Euclidean with detour factor for distant pairs
                    distances[sid_i][sid_j] = euc_dist * 1.3
            
            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(stop_ids)} stops...")
        
        return distances
    
    def _get_distance(self, stop1, stop2):
        """Get distance between two stops."""
        return self.distances[stop1][stop2]
    
    def _get_demand(self, stop_i, stop_j):
        """Get demand (trips) between two stops from the trips matrix."""
        try:
            idx_i = list(self.stops.keys()).index(stop_i)
            idx_j = list(self.stops.keys()).index(stop_j)
            demand = self.trips_matrix[idx_i, idx_j] + self.trips_matrix[idx_j, idx_i]
            max_demand = self.trips_matrix.max()
            return demand / max_demand if max_demand > 0 else 0
        except (ValueError, IndexError):
            return 0
    
    def _calculate_savings(self, depot=None, top_n_stops=None):
        """
        Calculate Clark-Wright savings - OPTIMIZED VERSION.
        
        Parameters:
        - depot: Depot stop ID
        - top_n_stops: Only consider top N stops by demand (speeds up for large networks)
        """
        stop_ids = list(self.stops.keys())
        
        # Filter to high-demand stops if requested
        if top_n_stops and top_n_stops < len(stop_ids):
            print(f"  Filtering to top {top_n_stops} stops by demand...")
            stop_demands = []
            for sid in stop_ids:
                try:
                    idx = list(self.stops.keys()).index(sid)
                    demand = self.trips_matrix[idx, :].sum() + self.trips_matrix[:, idx].sum()
                    stop_demands.append((demand, sid))
                except:
                    stop_demands.append((0, sid))
            
            stop_demands.sort(reverse=True)
            stop_ids = [sid for _, sid in stop_demands[:top_n_stops]]
        
        if depot is None:
            depot = stop_ids[0]
        
        print(f"Calculating Clark-Wright savings for {len(stop_ids)} stops...")
        savings = []
        
        for i in tqdm(range(len(stop_ids))):
            for j in range(i + 1, len(stop_ids)):
                stop_i = stop_ids[i]
                stop_j = stop_ids[j]
                
                if stop_i == depot or stop_j == depot:
                    continue
                
                d_depot_i = self._get_distance(depot, stop_i)
                d_depot_j = self._get_distance(depot, stop_j)
                d_i_j = self._get_distance(stop_i, stop_j)
                
                if d_depot_i == float('inf') or d_depot_j == float('inf') or d_i_j == float('inf'):
                    continue
                
                s = d_depot_i + d_depot_j - d_i_j
                demand = self._get_demand(stop_i, stop_j)
                weighted_savings = s * (1 + 0.5 * demand)
                
                savings.append((weighted_savings, stop_i, stop_j))
        
        savings.sort(reverse=True)
        print(f"  Calculated {len(savings)} savings values")
        return savings
    
    def design_routes(self, max_route_length=15, max_distance=30, depot=None, 
                     top_n_stops=None):
        """
        Design routes using Clark-Wright Savings algorithm.
        
        Parameters:
        - max_route_length: Maximum stops per route
        - max_distance: Maximum distance per route in km
        - depot: Depot stop ID (uses first stop if None)
        - top_n_stops: Only design routes for top N high-demand stops (speeds up large networks)
        """
        stop_ids = list(self.stops.keys())
        
        if depot is None:
            depot = stop_ids[0]
        
        print(f"\nDesigning routes with Clark-Wright algorithm...")
        print(f"  Max stops per route: {max_route_length}")
        print(f"  Max distance per route: {max_distance} km")
        
        savings = self._calculate_savings(depot, top_n_stops)
        
        # Initialize routes
        if top_n_stops and top_n_stops < len(stop_ids):
            # Only create routes for filtered stops
            active_stops = set()
            for _, si, sj in savings:
                active_stops.add(si)
                active_stops.add(sj)
            routes = {stop: [depot, stop, depot] for stop in active_stops}
            stop_to_route = {stop: stop for stop in active_stops}
        else:
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
                continue
            
            route_i_stops = routes[route_i]
            route_j_stops = routes[route_j]
            
            i_is_end = (route_i_stops[1] == stop_i or route_i_stops[-2] == stop_i)
            j_is_end = (route_j_stops[1] == stop_j or route_j_stops[-2] == stop_j)
            
            if not (i_is_end and j_is_end):
                continue
            
            merged = self._try_merge_routes(route_i_stops, route_j_stops, stop_i, stop_j, 
                                           max_route_length, max_distance)
            
            if merged:
                routes[route_i] = merged
                del routes[route_j]
                
                for stop in merged[1:-1]:
                    stop_to_route[stop] = route_i
                
                merges += 1
        
        print(f"  Completed {merges} route merges")
        
        self.routes = []
        for route_stops in routes.values():
            distance = self._calculate_route_distance(route_stops)
            if distance != float('inf'):  # Skip invalid routes
                self.routes.append(Route(route_stops, distance))
        
        print(f"\nCreated {len(self.routes)} routes")
        return self.routes
    
    def _try_merge_routes(self, route1, route2, stop_i, stop_j, max_length, max_distance):
        """Try to merge two routes."""
        r1 = route1[1:-1]
        r2 = route2[1:-1]
        
        if len(r1) + len(r2) > max_length:
            return None
        
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
            merged = [route1[0]] + merged + [route1[0]]
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
        """Calculate average wait time and route utilization."""
        print("\nCalculating performance metrics...")
        
        avg_wait_time = service_frequency_minutes / 2
        route_utilizations = []
        
        for route_idx, route in enumerate(self.routes):
            route_stop_ids = [s for s in route.stops if s in self.stops]
            
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
            
            total_trips = 0
            for i in stop_indices:
                for j in stop_indices:
                    if i != j:
                        total_trips += self.trips_matrix[i, j]
            
            daily_demand = total_trips / days_in_analysis if days_in_analysis else total_trips
            buses_per_day = (service_hours * 60) / service_frequency_minutes
            capacity = buses_per_day * bus_capacity
            utilization = (daily_demand / capacity * 100) if capacity > 0 else 0
            route_utilizations.append(min(utilization, 100))
        
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

if __name__ == "__main__":
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 18f8586 (add cacheing to measure_route_metrics)
    import sys
    args = sys.argv
    if len(args) < 2:
        print(f'Usage `{args[0]} <date>`')
        sys.exit(1)
    else:
        date = args[1]

<<<<<<< HEAD
    designer = TransitRouteDesigner(
        stops_csv_path='../../data/stops.csv',
        trips_npy_path=f'../../data/mlr_output_{date}.npy',
=======
    designer = TransitRouteDesigner(
        stops_csv_path='../../data/stops.csv',
        trips_npy_path='../../data/mlr_output_2021_11_18.npy',
>>>>>>> 748e956 (clarke_write metrics and implementation)
=======
    designer = TransitRouteDesigner(
        stops_csv_path='../../data/stops.csv',
        trips_npy_path=f'../../data/mlr_output_{date}.npy',
>>>>>>> 18f8586 (add cacheing to measure_route_metrics)
        city_name='London, Ontario, Canada',
        use_euclidean=True,
        cache_file='cache/distances_cache.pkl'
    )
    
    # For 2000 stops, only design routes for top 500 high-demand stops
    routes = designer.design_routes(
        max_route_length=25,
        max_distance=30,
        top_n_stops=500  # Focus on busiest stops
    )
    
    metrics = designer.calculate_metrics(
        service_frequency_minutes=20,
        service_hours=16,
        bus_capacity=50,
        days_in_analysis=30
    )
    
<<<<<<< HEAD
<<<<<<< HEAD
    routes_df = designer.export_routes(f'designed_routes/clarke_write-{date}')
=======
    routes_df = designer.export_routes('designed_routes.csv')
>>>>>>> 748e956 (clarke_write metrics and implementation)
=======
    routes_df = designer.export_routes(f'designed_routes/clarke_write-{date}')
>>>>>>> 18f8586 (add cacheing to measure_route_metrics)
    print("\n" + "="*60)
    print("DESIGNED ROUTES:")
    print("="*60)
    print(routes_df.to_string(index=False))