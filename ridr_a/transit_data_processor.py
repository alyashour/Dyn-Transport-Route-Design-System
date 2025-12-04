import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from util import *

ALL_EXPECTED_FEATURES = [
    'season', 
    'is_holiday', 
    'temp_high_c', 
    'temp_low_c', 
    'Population', 
    'Distance', 
    'Weather_Condition_Clear', 
    'Weather_Condition_Rainy', 
    'Weather_Condition_Snow'
]

class TransitDataProcessor:
    """Process and prepare transit data for modeling"""
    
    def __init__(self):
        self.stop_data = None
        self.weather_data = None
        self.population_data = None
        self.scaler = StandardScaler()
        self.n_stops = 0
        self.stop_id_to_idx = {}
        self.idx_to_stop_id = {}
        
    def load_data(self, trips_csv, weather_csv, stops_csv, population_csv):
        """Load all data files"""
        print("Loading data files...")
        
        # Load trips
        self.trips = pd.read_csv(trips_csv)
        print(f"Loaded {len(self.trips)} trips")
        
        # Load auxiliary data
        self.stop_data = pd.read_csv(stops_csv)
        self.weather_data = pd.read_csv(weather_csv)
        self.population_data = pd.read_csv(population_csv)
        
        # Create stop mappings
        unique_stops = sorted(self.stop_data['stop_id'].unique())
        self.n_stops = len(unique_stops)
        self.stop_id_to_idx = {sid: idx for idx, sid in enumerate(unique_stops)}
        self.idx_to_stop_id = {idx: sid for sid, idx in self.stop_id_to_idx.items()}
        
        print(f"Found {self.n_stops} unique stops")
        
    def engineer_features(self):
        """Create additional features"""
        print("Engineering features...")
        
        # Add season
        self.trips['season'] = self.trips['Month'].apply(get_season)
        
        # Add is_holiday
        self.trips['is_holiday'] = self.trips.apply(
            lambda row: is_holiday(row['Day'], row['Month'], row['Year']), axis=1
        ).astype(int)
        
        # Merge weather data
        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'])
        self.trips['Date'] = pd.to_datetime(
            self.trips[['Year', 'Month', 'Day']]
        )
        self.trips = self.trips.merge(self.weather_data, on='Date', how='left')
        
        # Merge population data
        self.trips = self.trips.merge(self.population_data, on='Year', how='left')
        
        # One-hot encode weather
        weather_dummies = pd.get_dummies(self.trips['Weather_Condition'], prefix='Weather_Condition')
        self.trips = pd.concat([self.trips, weather_dummies], axis=1)
        
        # Add stop coordinates
        stop_coords = self.stop_data.set_index('stop_id')[['stop_lat', 'stop_lon']]
        self.trips = self.trips.merge(
            stop_coords,
            left_on='Origin ID', 
            right_index=True,
            suffixes=('', '_origin')
        )
        self.trips.rename(columns={'stop_lat': 'origin_lat', 'stop_lon': 'origin_lon'}, inplace=True)
        
        self.trips = self.trips.merge(
            stop_coords,
            left_on='Destination ID',
            right_index=True,
            suffixes=('', '_dest')
        )
        self.trips.rename(columns={'stop_lat': 'dest_lat', 'stop_lon': 'dest_lon'}, inplace=True)
        
        # Calculate distance
        self.trips['Distance'] = haversine_distance(
            self.trips['origin_lat'], self.trips['origin_lon'],
            self.trips['dest_lat'], self.trips['dest_lon']
        )
        
        # Add stop indices
        self.trips['origin_idx'] = self.trips['Origin ID'].map(self.stop_id_to_idx)
        self.trips['dest_idx'] = self.trips['Destination ID'].map(self.stop_id_to_idx)
        
        # Drop unneeded columns
        self.trips = self.trips.drop(columns=['Origin ID', 'Destination ID', 'Date', 'Weather_Condition'])

        print("Feature engineering complete")
        
    def aggregate_daily_data(self):
        """Aggregate trips by date and route, computing target percentages"""
        print("Aggregating daily route statistics...")
        
        # Group by date and route
        date_cols = ['Year', 'Month', 'Day']
        route_cols = ['origin_idx', 'dest_idx']
        
        # Count trips per route per day
        route_counts = self.trips.groupby(date_cols + route_cols).size().reset_index(name='trip_count')
        
        # Get total trips per day
        daily_totals = self.trips.groupby(date_cols).size().reset_index(name='total_trips')
        
        # Merge and calculate percentages
        route_counts = route_counts.merge(daily_totals, on=date_cols)
        route_counts['route_percentage'] = route_counts['trip_count'] / route_counts['total_trips']
        
        # Merge back with features (take first occurrence per day for context features)
        feature_cols = ['season', 'is_holiday', 'temp_high_c', 'temp_low_c', 'Population', 'Distance'] + \
                      [col for col in self.trips.columns if col.startswith('Weather_')]
        
        daily_features = self.trips.groupby(date_cols + route_cols)[feature_cols].first().reset_index()
        
        self.daily_data = route_counts.merge(daily_features, on=date_cols + route_cols)
        
        print(f"Aggregated to {len(self.daily_data)} daily route observations")
        
        return self.daily_data
    
    def prepare_features(self, df: DataFrame):
        """
        Prepare feature matrix X from dataframe, ensuring all expected features 
        are present and correctly ordered.
        """
        
        # 1. Align the DataFrame columns
        
        # Iterate over the complete list of features the MLR model was trained on.
        for col in ALL_EXPECTED_FEATURES:
            # Check if the column is missing in the current DataFrame (df).
            if col not in df.columns:
                # If missing, add the column and fill it with 0.0. 
                df[col] = 0.0
                
        X = df[ALL_EXPECTED_FEATURES].values

        # X now has the correct shape (N_samples x 9) and order.
        return X
    
    def split_data(self, test_size=0.15, val_size=0.15):
        """Chronological train/val/test split"""
        print("Splitting data chronologically...")
        
        # Sort by date
        self.daily_data = self.daily_data.sort_values(['Year', 'Month', 'Day'])
        
        n = len(self.daily_data)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val
        
        train_data = self.daily_data.iloc[:n_train]
        val_data = self.daily_data.iloc[n_train:n_train+n_val]
        test_data = self.daily_data.iloc[n_train+n_val:]
        
        # Fit scaler on training data only
        X_train = self.prepare_features(train_data)
        self.scaler.fit(X_train)
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
