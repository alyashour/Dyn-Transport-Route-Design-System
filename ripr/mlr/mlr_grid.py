# ============================================================================
# MODEL 1: GRID OF MLR MODELS
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

from util import *

class MLRGrid:
    """Grid of Multiple Linear Regression models, one per route"""
    
    def __init__(self, n_stops):
        self.n_stops = n_stops
        self.models = {}  # Key: (origin_idx, dest_idx), Value: LinearRegression model
        self.min_samples = 10  # Minimum samples required to train a route model
        
    def fit(self, train_data, scaler):
        """Train individual MLR models for each route with sufficient data"""
        print(f"\nTraining MLR Grid...")
        
        # Group by route
        route_groups = train_data.groupby(['origin_idx', 'dest_idx'])
        
        trained_routes = 0
        for (origin, dest), group in route_groups:
            if len(group) >= self.min_samples:
                X = scaler.transform(self._prepare_features(group))
                y = group['route_percentage'].values
                
                model = LinearRegression()
                model.fit(X, y)
                self.models[(origin, dest)] = model
                trained_routes += 1
        
        print(f"Trained {trained_routes} route-specific MLR models")
        
    def _prepare_features(self, df):
        """Extract features from dataframe"""
        feature_cols = ['season', 'is_holiday', 'temp_high_c', 'temp_low_c', 'Population', 'Distance', 'Weather_Condition_Clear', 'Weather_Condition_Rainy', 'Weather_Condition_Snowy']
        return df[feature_cols].values
    
    def predict(self, test_data, scaler):
        """Predict route percentages and construct probability matrix"""
        print("\nPredicting with MLR Grid...")
        
        # Group test data by date
        date_cols = ['Year', 'Month', 'Day']
        date_groups = test_data.groupby(date_cols)
        
        predictions = []
        
        for date_key, group in date_groups:
            # Create empty matrix for this date
            route_matrix = np.zeros((self.n_stops, self.n_stops))
            
            # Predict for each route in the group
            for _, row in group.iterrows():
                origin = int(row['origin_idx'])
                dest = int(row['dest_idx'])
                
                if (origin, dest) in self.models:
                    X = scaler.transform(self._prepare_features(pd.DataFrame([row])))
                    pred = self.models[(origin, dest)].predict(X)[0]
                    route_matrix[origin, dest] = max(0, pred)  # Clip negative predictions
                else:
                    # Use mean percentage for unseen routes
                    route_matrix[origin, dest] = 0.0001  # Small non-zero value

            # Disabling softmax because it's cooked
            """ 
            # Apply softmax to ensure probabilities sum to 1
            route_matrix_flat = route_matrix.flatten()
            probs = self._softmax(route_matrix_flat)
            route_matrix = probs.reshape(self.n_stops, self.n_stops)
            """
            
            predictions.append({
                'Date': date_key,
                'matrix': route_matrix
            })
        
        return predictions
    
    def _softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def save(self, filepath):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
