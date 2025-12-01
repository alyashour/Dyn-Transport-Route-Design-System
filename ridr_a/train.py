import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transit_data_processor import TransitDataProcessor
from util import *
from mlr_grid import MLRGrid

SKIP_CONFIRMATION = True

class TransitDataset(Dataset):
    """PyTorch dataset for transit data"""
    
    def __init__(self, data, scaler, n_stops):
        self.n_stops = n_stops
        
        # Prepare features
        feature_cols = ['season', 'is_holiday', 'temp_high_c', 'temp_low_c', 'Population', 'Distance'] + \
                      [col for col in data.columns if col.startswith('Weather_')]
        self.X = torch.FloatTensor(scaler.transform(data[feature_cols].values))
        
        # Prepare targets (route indices and percentages)
        self.origin_idx = torch.LongTensor(data['origin_idx'].values)
        self.dest_idx = torch.LongTensor(data['dest_idx'].values)
        self.route_pct = torch.FloatTensor(data['route_percentage'].values)
        
        # Date information for grouping
        self.dates = data[['Year', 'Month', 'Day']].values
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'origin': self.origin_idx[idx],
            'dest': self.dest_idx[idx],
            'route_pct': self.route_pct[idx],
            'Date': tuple(self.dates[idx])
        }

class MLPModel(nn.Module):
    """MLP that predicts route probability matrix"""
    
    def __init__(self, n_features, n_stops, hidden_dims=[32, 32]):
        super().__init__()
        self.n_stops = n_stops
        
        # Build MLP layers
        layers = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        # Output layer: predicts logits for all possible routes
        layers.append(nn.Linear(in_dim, n_stops * n_stops))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_features)
        Returns:
            route_probs: (batch_size, n_stops, n_stops) - probability matrix
        """
        logits = self.network(x)  # (batch_size, n_stops * n_stops)
        logits = logits.view(-1, self.n_stops, self.n_stops)
        
        # Apply softmax over all routes
        logits_flat = logits.view(logits.size(0), -1)
        probs_flat = F.softmax(logits_flat, dim=1)
        probs = probs_flat.view(-1, self.n_stops, self.n_stops)
        
        return probs

# ============================================================================
# EVALUATION
# ============================================================================

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_models(mlr_grid, test_data, scaler):
    """Evaluate the MLR Grid model on the test set and print overall metrics."""
    print("\n" + "="*80)
    print("EVALUATING MLR GRID MODEL")
    print("="*80)
    
    # --- Step 1: Get Predictions ---
    # The mlr_grid.predict function returns a list of dictionaries, 
    # where each dict contains the date and the full predicted route matrix.
    mlr_predictions = mlr_grid.predict(test_data, scaler)
    
    # --- Step 2: Prepare True and Predicted Values ---
    
    # Extract true values from the test data
    true_percentages = test_data['route_percentage'].values
    
    # Flatten the prediction matrices and align them with the test_data
    
    # Create a dictionary for quick lookup of the predicted matrix by date
    pred_matrix_map = {pred['Date']: pred['matrix'] for pred in mlr_predictions}
    
    predicted_percentages = []
    
    # Iterate through each row of the test data
    for index, row in test_data.iterrows():
        date_key = tuple(row[['Year', 'Month', 'Day']].values)
        origin_idx = int(row['origin_idx'])
        dest_idx = int(row['dest_idx'])
        
        # Look up the predicted matrix for that date
        matrix = pred_matrix_map.get(date_key)
        
        if matrix is not None:
            # Extract the predicted percentage for the specific route (origin, dest)
            pred_pct = matrix[origin_idx, dest_idx]
            predicted_percentages.append(pred_pct)
        else:
            # Should not happen if data is correctly aligned, but necessary for safety
            predicted_percentages.append(0.0) 
            
    # Convert to NumPy array for metric calculation
    predicted_percentages = np.array(predicted_percentages)
    
    # --- Step 3: Calculate Metrics ---
    
    # The MLR grid uses Softmax, which is often optimized for classification 
    # (probability distribution), but we use regression metrics for percentage error.
    
    # Mean Squared Error (MSE): Punishes larger errors more heavily.
    mse = mean_squared_error(true_percentages, predicted_percentages)
    
    # Mean Absolute Error (MAE): Easier to interpret, represents average error magnitude.
    mae = mean_absolute_error(true_percentages, predicted_percentages)
    
    # --- Step 4: Print Results ---
    print("\nMLR Grid Performance Metrics:")
    print(f"  -> Mean Squared Error (MSE): {mse:.6f}")
    print(f"  -> Mean Absolute Error (MAE): {mae:.6f}")
    
    return {
        'mlr_mse': mse,
        'mlr_mae': mae
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Configuration
    TRIPS_CSV = 'data/trips_data.csv'
    WEATHER_CSV = 'data/weather.csv'
    STOPS_CSV = 'data/stop_data.csv'
    POPULATION_CSV = 'data/city_population.csv'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Step 1: Load and preprocess data
    processor = TransitDataProcessor()
    processor.load_data(TRIPS_CSV, WEATHER_CSV, STOPS_CSV, POPULATION_CSV)
    processor.engineer_features()
    processor.aggregate_daily_data()

    # clear space (4GB of cache is wild work gang)
    if DEVICE == 'cuda':
        print('Clearing Preprocess CUDA cache')
        torch.cuda.empty_cache()

    if not SKIP_CONFIRMATION:
        # Step 1.1: Confirm Columns
        print(f'Columns:\n {processor.trips.columns}')
        if input('Confirm (y/n): ') not in ['y', 'Y']:
            print('Canceling')
            raise "Canceled"
        
        # Step 1.2: Head data
        print(f'Head:\n{processor.trips.head()}')
        if input('Confirm (y/n): ') not in ['y', 'Y']:
            print('Canceling')
            raise "Canceled"
    
    # Step 2: Split data
    train_data, val_data, test_data = processor.split_data()
    
    # Step 3: Train MLR Grid
    mlr_grid = MLRGrid(processor.n_stops)
    mlr_grid.fit(train_data, processor.scaler)
    mlr_grid.save('models/mlr_grid_model_latest.pkl')

    # clear space
    if DEVICE == 'cuda':
        print('Clearing MLR CUDA cache')
        torch.cuda.empty_cache()
    
    # clear space
    if DEVICE == 'cuda':
        print('Clearing datasets step CUDA cache')
        torch.cuda.empty_cache()

    # Step 7: Evaluate all models
    results = evaluate_models(mlr_grid, test_data, processor.scaler)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nModels saved:")
    print("- models/mlr_grid_model_latest.pkl")
    
    return processor, mlr_grid, results

if __name__ == "__main__":
    processor, mlr_grid, results = main()
    print(results)