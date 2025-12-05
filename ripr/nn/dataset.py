import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from util import get_season, is_holiday

class TransitDataset(Dataset):
    def __init__(self, h5_file):
        super().__init__()
        self.h5_file = h5_file
        
        # 1. Initialization
        # We only open the file briefly to get the length and number of stops.
        # absolutely NO data is loaded into RAM here.
        with h5py.File(h5_file, 'r') as f:
            self.num_stops = f['stops'].shape[0]
            self.num_days = f['dates'].shape[0]

        # Define Feature Dimensions for One-Hot Encoding
        self.n_months = 12     # 1-12
        self.n_days = 31       # 1-31
        self.n_dows = 7        # 0-6
        self.n_seasons = 4     # 0-3
        self.n_weather = 3     # 0-2 (Clear, Rain, Snow)
        
    def __len__(self):
        return self.num_days

    def _get_one_hot(self, value, num_classes, offset=0):
        """Helper to create one-hot vector."""
        idx = value - offset
        idx = max(0, min(idx, num_classes - 1))
        
        vec = torch.zeros(num_classes, dtype=torch.float32)
        vec[idx] = 1.0
        return vec

    def __getitem__(self, idx):
        # Open file fresh for every item. 
        # This allows multiple workers to access the file safely without pickling issues.
        with h5py.File(self.h5_file, 'r') as f:
            
            # ---------------------------------------------------------
            # 1. READ METADATA (From Disk)
            # ---------------------------------------------------------
            # We access the specific index [idx] directly from the H5 dataset object
            
            # dates cols: [Year-2000, Month, Day, DayOfWeek, WeatherIdx]
            dates_row = f['dates'][idx] 
            
            # temps cols: [High, Low]
            temps_row = f['temps'][idx]
            
            # day_trips cols: [Total_Count, Start_Idx, End_Idx]
            pointers_row = f['day_trips'][idx]

            # ---------------------------------------------------------
            # 2. FEATURE ENGINEERING
            # ---------------------------------------------------------
            year = int(dates_row[0]) + 2000
            month = int(dates_row[1])
            day = int(dates_row[2])
            dow = int(dates_row[3])
            weather = int(dates_row[4])
            
            # Derive Missing Features
            season = get_season(month)
            holiday = is_holiday(day, month, year)
            
            # Build One-Hot Vectors
            oh_month = self._get_one_hot(month, self.n_months, offset=1)
            oh_day = self._get_one_hot(day, self.n_days, offset=1)
            oh_dow = self._get_one_hot(dow, self.n_dows, offset=0)
            oh_season = self._get_one_hot(season, self.n_seasons, offset=0)
            oh_weather = self._get_one_hot(weather, self.n_weather, offset=0)
            
            # Scalar Features
            feat_holiday = torch.tensor([float(holiday)], dtype=torch.float32)
            feat_temps = torch.from_numpy(temps_row.astype(np.float32))
            
            x = torch.cat([
                oh_month, oh_day, oh_dow, oh_season, oh_weather, 
                feat_holiday, feat_temps
            ])

            # ---------------------------------------------------------
            # 3. TARGET OUTPUT
            # ---------------------------------------------------------
            target = torch.zeros(self.num_stops * self.num_stops, dtype=torch.float32)
            
            total_count = pointers_row[0]
            start_ptr = pointers_row[1]
            end_ptr = pointers_row[2]
            
            if total_count > 0:
                # Read specific slice of trips from disk
                # f['trips'] is the dataset on disk, slicing it reads only that chunk
                trips_slice = f['trips'][start_ptr : end_ptr]
                
                origins = trips_slice[:, 1].astype(np.int64)
                dests = trips_slice[:, 2].astype(np.int64)
                counts = trips_slice[:, 3].astype(np.float32)
                
                flat_indices = origins * self.num_stops + dests
                
                target[flat_indices] = torch.from_numpy(counts)
                target = target / float(total_count)
            
        return x, target

if __name__ == "__main__":
    ds = TransitDataset('dataset.h5')
    print(f"Dataset Length: {len(ds)} days")
    
    x, y = ds[100] 
    print(f"Input Shape: {x.shape}")
    print(f"Target Shape: {y.shape}")
