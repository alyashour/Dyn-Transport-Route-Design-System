import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
import numpy as np
from util import get_season, is_holiday
from tqdm import tqdm

# Feature Dimensions
N_MONTHS = 12     # 0-11 (User handled 0-indexing)
N_DAYS = 31       # 0-30
N_DOWS = 7        # 0-6
N_SEASONS = 4     # 0-3
N_WEATHER = 3     # 0-2 (Clear, Rain, Snow)

def encode_features(month, day, dow, season, weather, holiday, temps):
    """
    Encodes raw feature values into a concatenated tensor.
    Shared by both the Dataset (training) and Inference scripts.
    
    Args:
        month (int): 0-11
        day (int): 0-30
        dow (int): 0-6
        season (int): 0-3
        weather (int): 0-2
        holiday (bool/int/float): 0 or 1
        temps (list/tensor/array): [High, Low]
    
    Returns:
        torch.Tensor: Concatenated feature vector
    """
    # Build One-Hot Vectors using torch's native method
    # We create tensors, cast to Long (int64) for one_hot, then Float for the network
    oh_month = F.one_hot(torch.tensor(month - 1, dtype=torch.long), num_classes=N_MONTHS).float()
    oh_day = F.one_hot(torch.tensor(day - 1, dtype=torch.long), num_classes=N_DAYS).float()
    oh_dow = F.one_hot(torch.tensor(dow, dtype=torch.long), num_classes=N_DOWS).float()
    oh_season = F.one_hot(torch.tensor(season, dtype=torch.long), num_classes=N_SEASONS).float()
    oh_weather = F.one_hot(torch.tensor(weather, dtype=torch.long), num_classes=N_WEATHER).float()
    
    # Scalar Features - Handle various input types (Tensor vs scalar)
    if isinstance(holiday, torch.Tensor):
        feat_holiday = holiday.float().view(1)
    else:
        feat_holiday = torch.tensor([float(holiday)], dtype=torch.float32)
        
    if isinstance(temps, torch.Tensor):
        feat_temps = temps.float()
    else:
        feat_temps = torch.tensor(temps, dtype=torch.float32)
    
    # Concatenate all features
    return torch.cat([
        oh_month, oh_day, oh_dow, oh_season, oh_weather, 
        feat_holiday, feat_temps
    ])

class ShotgunDataset(Dataset):
    def __init__(self, h5_file):
        super().__init__()
        self.h5_file = h5_file
        
        # 1. Initialization
        # We only open the file briefly to get the length and number of stops.
        # absolutely NO data is loaded into RAM here.
        with h5py.File(h5_file, 'r') as f:
            self.num_stops = f['stops'].shape[0]
            self.num_days = f['dates'].shape[0]

    def __len__(self):
        return self.num_days

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
            
            # Use shared encoding function
            x = encode_features(month, day, dow, season, weather, holiday, temps_row)

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
    ds = ShotgunDataset('dataset.h5')
    print(f"Dataset Length: {len(ds)} days")
    
    x, y = ds[100] 
    print(f"Input Shape: {x.shape}")
    print(f"Target Shape: {y.shape}")

    for i in tqdm(range(len(ds))):
        ds[i]

    print("No failed retrievals!")
