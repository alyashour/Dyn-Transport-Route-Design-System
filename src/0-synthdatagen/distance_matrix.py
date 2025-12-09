"""
Distance Matrix Generator for London, ON Transit Stops
Uses OSMNx to calculate shortest path distances between stops on the road network.

Requirements:
    pip install osmnx networkx pandas numpy tqdm scikit-learn

Usage:
    python generate_distance_matrix.py

Notes:
    - This script can take a LONG time to run with ~1900 stops (1900² = 3.6M pairs)
    - Consider using a subset of stops or the --sample flag for testing
    - The script saves progress periodically and can resume from where it left off
"""

import os
import argparse
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from tqdm import tqdm
import pickle
from datetime import datetime

# Configure OSMNx
ox.settings.log_console = False
ox.settings.use_cache = True


def load_stops(csv_path: str, sample_size: int = None) -> pd.DataFrame:
    """Load stop data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ['stop_id', 'stop_lat', 'stop_lon']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Remove rows with missing coordinates
    df = df.dropna(subset=['stop_lat', 'stop_lon'])
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} stops for testing")
    
    print(f"Loaded {len(df)} stops")
    return df


def download_road_network(stops_df: pd.DataFrame, buffer_km: float = 2.0) -> nx.MultiDiGraph:
    """
    Download the road network for the area covering all stops.
    
    Args:
        stops_df: DataFrame with stop_lat and stop_lon columns
        buffer_km: Buffer around the bounding box in kilometers
    """
    # Calculate bounding box with buffer
    min_lat = stops_df['stop_lat'].min()
    max_lat = stops_df['stop_lat'].max()
    min_lon = stops_df['stop_lon'].min()
    max_lon = stops_df['stop_lon'].max()
    
    # Add buffer (roughly convert km to degrees)
    buffer_deg = buffer_km / 111  # ~111 km per degree
    
    north = max_lat + buffer_deg
    south = min_lat - buffer_deg
    east = max_lon + buffer_deg
    west = min_lon - buffer_deg
    
    print(f"Downloading road network for London, ON...")
    print(f"Bounding box: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")
    
    # Download the drivable road network
    # Using 'drive' network type for bus routes
    # Use explicit parameter names for compatibility
    G = ox.graph_from_bbox(
        north=north,
        south=south,
        east=east,
        west=west,
        network_type='drive',
        simplify=True
    )
    
    print(f"Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G


def get_nearest_nodes(G: nx.MultiDiGraph, stops_df: pd.DataFrame) -> tuple[nx.MultiDiGraph, dict]:
    """Find the nearest network node for each stop."""
    print("Finding nearest network nodes for each stop...")
    
    # Project the graph to a local UTM coordinate system
    G_projected = ox.project_graph(G)
    
    # Get the CRS from the projected graph
    crs = G_projected.graph['crs']
    
    # Convert stop lat/lon to the projected coordinate system
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    # Transform coordinates (lon, lat) -> (x, y)
    lons = stops_df['stop_lon'].values
    lats = stops_df['stop_lat'].values
    xs, ys = transformer.transform(lons, lats)
    
    # Find nearest nodes using projected coordinates
    nearest_nodes = ox.nearest_nodes(G_projected, xs, ys)
    
    # Create mapping from stop_id to node
    stop_to_node = dict(zip(stops_df['stop_id'], nearest_nodes))
    
    return G_projected, stop_to_node


def calculate_distance_matrix(
    G: nx.MultiDiGraph,
    stops_df: pd.DataFrame,
    stop_to_node: dict,
    output_path: str,
    checkpoint_interval: int = 100
) -> pd.DataFrame:
    """
    Calculate the distance matrix between all stops.
    
    Uses shortest path distances on the road network.
    Saves checkpoints periodically to allow resuming.
    """
    stop_ids = stops_df['stop_id'].tolist()
    n_stops = len(stop_ids)
    
    print(f"Calculating {n_stops}x{n_stops} = {n_stops**2:,} distance pairs...")
    
    # Check for existing checkpoint
    checkpoint_path = output_path.replace('.csv', '_checkpoint.pkl')
    start_idx = 0
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            distances = checkpoint['distances']
            start_idx = checkpoint['completed_rows']
            print(f"Resuming from checkpoint: {start_idx}/{n_stops} rows completed")
    else:
        # Initialize distance matrix with infinity
        distances = np.full((n_stops, n_stops), np.inf)
        np.fill_diagonal(distances, 0)  # Distance to self is 0
    
    # Add edge lengths if not present
    # In newer OSMNx versions, lengths are added automatically, but let's verify
    sample_edge = list(G.edges(data=True))[0]
    if 'length' not in sample_edge[2]:
        # Try the new location for add_edge_lengths
        try:
            G = ox.distance.add_edge_lengths(G)
        except AttributeError:
            # Calculate lengths manually using the projected coordinates
            for u, v, data in G.edges(data=True):
                if 'length' not in data:
                    # Get node coordinates
                    u_data = G.nodes[u]
                    v_data = G.nodes[v]
                    # Calculate Euclidean distance (graph is projected, so this is in meters)
                    dx = u_data['x'] - v_data['x']
                    dy = u_data['y'] - v_data['y']
                    data['length'] = (dx**2 + dy**2)**0.5
    
    # Calculate shortest paths from each stop
    for i in tqdm(range(start_idx, n_stops), desc="Computing distances"):
        origin_stop = stop_ids[i]
        origin_node = stop_to_node[origin_stop]
        
        try:
            # Calculate shortest paths from this node to all other nodes
            lengths = nx.single_source_dijkstra_path_length(
                G, origin_node, weight='length'
            )
            
            # Fill in the distances for this row
            for j, dest_stop in enumerate(stop_ids):
                dest_node = stop_to_node[dest_stop]
                if dest_node in lengths:
                    distances[i, j] = lengths[dest_node]
                    
        except nx.NetworkXError as e:
            print(f"Warning: Could not compute paths from {origin_stop}: {e}")
        
        # Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'distances': distances,
                    'completed_rows': i + 1
                }, f)
            print(f"Checkpoint saved at row {i + 1}")
    
    # Create DataFrame with stop IDs as index and columns
    distance_df = pd.DataFrame(
        distances,
        index=stop_ids,
        columns=stop_ids
    )
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return distance_df


def main():
    parser = argparse.ArgumentParser(
        description='Generate distance matrix for transit stops using OSMNx'
    )
    parser.add_argument(
        '--input', '-i',
        default='stop_data.csv',
        help='Path to input CSV file with stop data'
    )
    parser.add_argument(
        '--output', '-o',
        default='distance_matrix.csv',
        help='Path to output CSV file for distance matrix'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=None,
        help='Number of stops to sample (for testing). Use a small number like 50 for testing.'
    )
    parser.add_argument(
        '--buffer', '-b',
        type=float,
        default=2.0,
        help='Buffer around stops bounding box in km (default: 2.0)'
    )
    parser.add_argument(
        '--checkpoint-interval', '-c',
        type=int,
        default=100,
        help='Save checkpoint every N rows (default: 100)'
    )
    parser.add_argument(
        '--output-format',
        choices=['csv', 'parquet', 'both'],
        default='csv',
        help='Output format (default: csv). Parquet is more efficient for large matrices.'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Distance Matrix Generator for London, ON Transit")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load stops
    stops_df = load_stops(args.input, args.sample)
    
    # Download road network
    G = download_road_network(stops_df, args.buffer)
    
    # Find nearest nodes (also projects the graph)
    G, stop_to_node = get_nearest_nodes(G, stops_df)
    
    # Calculate distance matrix
    distance_df = calculate_distance_matrix(
        G, stops_df, stop_to_node,
        args.output,
        args.checkpoint_interval
    )
    
    # Save results
    print(f"\nSaving distance matrix...")
    
    if args.output_format in ['csv', 'both']:
        csv_path = args.output
        distance_df.to_csv(csv_path)
        print(f"Saved CSV: {csv_path}")
        
        # Also save a summary
        summary_path = csv_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Distance Matrix Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Number of stops: {len(distance_df)}\n")
            f.write(f"Total pairs: {len(distance_df)**2:,}\n")
            f.write(f"\nDistance statistics (meters):\n")
            
            # Flatten and remove inf/zero values for stats
            flat = distance_df.values.flatten()
            valid = flat[(flat > 0) & (flat < np.inf)]
            
            if len(valid) > 0:
                f.write(f"  Min: {valid.min():.1f}\n")
                f.write(f"  Max: {valid.max():.1f}\n")
                f.write(f"  Mean: {valid.mean():.1f}\n")
                f.write(f"  Median: {np.median(valid):.1f}\n")
            else:
                f.write("  No valid distances found\n")
            
            unreachable = np.sum(flat == np.inf)
            f.write(f"\nUnreachable pairs: {unreachable:,} ({unreachable/len(flat)*100:.2f}%)\n")
        print(f"Saved summary: {summary_path}")
    
    if args.output_format in ['parquet', 'both']:
        parquet_path = args.output.replace('.csv', '.parquet')
        distance_df.to_parquet(parquet_path)
        print(f"Saved Parquet: {parquet_path}")
    
    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Done!")


if __name__ == '__main__':
    main()