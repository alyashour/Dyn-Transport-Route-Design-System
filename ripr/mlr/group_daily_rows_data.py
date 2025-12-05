import pandas as pd
import numpy as np

def group_daily_data(input_csv, output_csv):
    """
    Group transit data by day only, dropping route-specific columns.
    This creates one row per date for inference.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
    """
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop columns we don't need for inference
    columns_to_drop = ['trip_count', 'origin_idx', 'dest_idx', 'route_percentage', 'total_trips', 'Distance']
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    
    if existing_drops:
        df = df.drop(columns=existing_drops)
        print(f"Dropped columns: {existing_drops}")
    
    # Group by date (Year, Month, Day) and take first value for each group
    # Since weather, population, etc. are the same for all routes on a given day
    date_cols = ['Year', 'Month', 'Day']
    
    print("\nGrouping by date...")
    grouped_df = df.groupby(date_cols).first().reset_index()
    
    print(f"\nOriginal dataset: {len(df)} rows")
    print(f"Grouped dataset: {len(grouped_df)} rows (one per date)")
    print(f"Columns: {grouped_df.columns.tolist()}")
    
    # Save to CSV
    print(f"\nSaving to {output_csv}...")
    grouped_df.to_csv(output_csv, index=False)
    print("Done!")
    
    return grouped_df

if __name__ == "__main__":
    # Configuration
    INPUT_CSV = "data/processed_combined_data.csv"
    OUTPUT_CSV = "data/daily_grouped_inference_data.csv"
    
    # Process the data
    df = group_daily_data(INPUT_CSV, OUTPUT_CSV)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['Year'].min()}/{df['Month'].min()}/{df['Day'].min()} to {df['Year'].max()}/{df['Month'].max()}/{df['Day'].max()}")
    print(f"\nColumns in final dataset:")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nLast few rows:")
    print(df.tail(10))