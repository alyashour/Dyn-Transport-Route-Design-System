from pathlib import Path 

import numpy as np
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in km"""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def get_season(month):
    """Convert month to season (0=Winter, 1=Spring, 2=Summer, 3=Fall)"""
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3

def is_holiday(day, month, year):
    """Check if date is a holiday (weekend or winter break)"""
    date = datetime(year, month, day)
    
    # Weekend
    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return True
    
    # Winter break (Dec 23 - Jan 7)
    if (month == 12 and day >= 23) or (month == 1 and day <= 7):
        return True
    
    return False

def get_root(path: Path = Path(__file__).resolve()):
    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Not inside a git repo")