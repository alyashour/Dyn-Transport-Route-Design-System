import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Configuration
H5_FILE = 'dataset.h5'

def plot_heatmap(grid, title):
    """
    Helper to plot the matrix with sparsity stats.
    """
    # 1. Calculate Sparsity
    total_cells = grid.size
    zero_cells = np.sum(grid == 0)
    sparsity_pct = (zero_cells / total_cells) * 100
    
    print(f"--- {title} ---")
    print(f"Grid Size: {grid.shape}")
    print(f"Sparsity: {sparsity_pct:.4f}% of the grid is empty (0 trips).")
    
    # 2. Plot
    plt.figure(figsize=(10, 8))
    
    # Create a copy for plotting to handle log(0)
    plot_data = grid.copy()
    plot_data[plot_data == 0] = np.nan 
    
    plt.imshow(grid + 1, cmap='inferno', norm=LogNorm(), interpolation='nearest')
    plt.colorbar(label='Trip Count (Log Scale)')
    plt.title(title)
    plt.xlabel("Destination Stop Index")
    plt.ylabel("Origin Stop Index")
    plt.tight_layout()
    plt.show()

def plot_histogram(grid, title="Trip Count Distribution"):
    """
    Plots a histogram of non-zero trip counts to visualize the power law distribution.
    """
    print(f"Generating histogram for: {title}...")
    
    # Flatten and filter for non-zero trips only
    non_zero_trips = grid[grid > 0].flatten()
    
    if len(non_zero_trips) == 0:
        print("No trips found to histogram.")
        return

    plt.figure(figsize=(10, 6))
    
    # Use log scale on Y-axis because low-count routes are vastly more common than high-count ones
    plt.hist(non_zero_trips, bins=100, log=True, color='#FF5733', alpha=0.7, edgecolor='black')
    
    plt.title(f"Histogram: {title}")
    plt.xlabel("Number of Trips on Route")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

def visualize_trips():
    with h5py.File(H5_FILE, 'r') as f:
        # Load Dimensions
        num_stops = f['stops'].shape[0] # type: ignore
        
        # Load Date Metadata
        day_trips = f['day_trips'][:]
        
        # Reconstruct Datetime index
        years = day_trips[:, 0] + 2000
        months = day_trips[:, 1]
        days = day_trips[:, 2]
        dates = pd.to_datetime({'year': years, 'month': months, 'day': days})
        
        min_date = dates.min().date()
        max_date = dates.max().date()
        print(f"Loaded dataset covering {min_date} to {max_date}")

        # ---------------------------------------------------------
        # SCENARIO 1: ENTIRE TIME SPAN
        # ---------------------------------------------------------
        print("\nAggregating ENTIRE dataset...")
        
        all_trips = f['trips'][:] 
        
        full_grid = np.zeros((num_stops, num_stops), dtype=np.float32)
        np.add.at(full_grid, (all_trips[:, 1], all_trips[:, 2]), all_trips[:, 3])
        
        plot_heatmap(full_grid, "Full Time Span OD Grid")
        plot_histogram(full_grid, "Full Time Span Trip Counts")

        # ---------------------------------------------------------
        # SCENARIO 2: USER INPUT LOOP
        # ---------------------------------------------------------
        while True:
            print("\n" + "="*40)
            print(f"Data Available: {min_date} to {max_date}")
            print("Enter date range to visualize (or leave blank to exit).")
            
            start_str = input("Start Date (YYYY-MM-DD): ").strip()
            if not start_str:
                print("Exiting...")
                break
                
            end_str = input("End Date   (YYYY-MM-DD): ").strip()
            if not end_str:
                print("Exiting...")
                break
            
            try:
                # Convert Input
                start_date = pd.to_datetime(start_str)
                end_date = pd.to_datetime(end_str)
                
                print(f"\nAggregating range: {start_date.date()} to {end_date.date()}...")
                
                # 1. Find rows in day_trips corresponding to this range
                mask = (dates >= start_date) & (dates <= end_date)
                selected_days = day_trips[mask]
                
                if len(selected_days) == 0:
                    print(f"No data found between {start_str} and {end_str}.")
                    continue

                # 2. Determine range pointers
                range_start_ptr = selected_days[0, 4]  # Start index of first day
                range_end_ptr = selected_days[-1, 5]   # End index of last day
                
                # 3. Load Slice
                range_trips = f['trips'][range_start_ptr : range_end_ptr]
                
                if len(range_trips) == 0:
                    print("Date range exists, but contains 0 trips.")
                    continue

                # 4. Aggregate
                range_grid = np.zeros((num_stops, num_stops), dtype=np.float32)
                np.add.at(range_grid, (range_trips[:, 1], range_trips[:, 2]), range_trips[:, 3])
                
                plot_heatmap(range_grid, f"OD Grid ({start_date.date()} - {end_date.date()})")
                
            except ValueError:
                print("Error: Invalid date format. Please use YYYY-MM-DD.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    visualize_trips()
