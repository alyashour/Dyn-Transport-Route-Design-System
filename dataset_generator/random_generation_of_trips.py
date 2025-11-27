import pandas as pd
import numpy as np
import datetime

# --- CONFIGURATION ---
INITIAL_LONDON_POP = 422000
BASE_RIDERSHIP_RATE = 0.07  

ZONE_WEIGHTS = {
    'Residential': {'pop': 100, 'attr': 10},
    'University':  {'pop': 200, 'attr': 1000}, 
    'Hub':         {'pop': 50,  'attr': 400},
    'Downtown':    {'pop': 50,  'attr': 600},
}

# --- DATA PREP ---
def load_and_prep_data(filename):
    df = pd.read_csv(filename)
    
    def get_stop_type(row):
        name = str(row['stop_name']).lower()
        zone = str(row['Name'])
        if any(x in name for x in ['western', 'university', 'natural science', 'alumni', 'fanshawe']): return 'University'
        if any(x in name for x in ['masonville', 'white oaks', 'argyle', 'westmount']): return 'Hub'
        if zone == 'Central_London' or 'downtown' in name: return 'Downtown'
        return 'Residential'

    df['type'] = df.apply(get_stop_type, axis=1)
    df['base_pop'] = df['type'].map(lambda x: ZONE_WEIGHTS.get(x, ZONE_WEIGHTS['Residential'])['pop'])
    df['base_attr'] = df['type'].map(lambda x: ZONE_WEIGHTS.get(x, ZONE_WEIGHTS['Residential'])['attr'])
    return df

# --- DYNAMIC FACTORS ---
def get_academic_multiplier(date_obj):
    year, m, d = date_obj.year, date_obj.month, date_obj.day
    current = datetime.datetime(year, m, d)
    
    #define Key Periods
    f_exam_start = datetime.datetime(year, 12, 8)
    f_exam_end = datetime.datetime(year, 12, 22)
    winter_break_start = datetime.datetime(year, 12, 23)
    #winter break usually goes into the next year (Jan 7)
    #we handle Jan 1-7 separately in the month check below
    
    w_exam_start = datetime.datetime(year, 4, 10)
    w_exam_end = datetime.datetime(year, 4, 30)

    #summer (May-Aug) -> LOW
    if 5 <= m <= 8: return 0.6
    
    #winter break (Jan 1 - Jan 7) -> LOW
    if m == 1 and d <= 7: return 0.6
    
    #winter break (Dec 23 - Dec 31) -> LOW
    if m == 12 and d >= 23: return 0.6

    #fall Ramp Up (Nov 13 - Dec 8) -> HIGH
    delta_f = (f_exam_start - current).days
    if 0 < delta_f <= 25: 
        # Linearly interpolate from 1.3 to 1.6
        return 1.3 + (0.3 * (1 - (delta_f / 25.0)))
        
    #fall Exam Decay (Dec 8 - Dec 22) -> DROPPING
    if f_exam_start <= current <= f_exam_end:
        return 1.3 - (0.7 * ((current - f_exam_start).days / (f_exam_end - f_exam_start).days))

    #winter Ramp Up (Mar 16 - Apr 10) -> HIGH
    delta_w = (w_exam_start - current).days
    if 0 < delta_w <= 25: 
        return 1.3 + (0.3 * (1 - (delta_w / 25.0)))
        
    #winter Exam Decay (Apr 10 - Apr 30) -> DROPPING
    if w_exam_start <= current <= w_exam_end:
        return 1.3 - (0.7 * ((current - w_exam_start).days / (w_exam_end - w_exam_start).days))
        
    return 1.3

def calculate_daily_trips(date_obj, weather, current_pop):
    N = current_pop * BASE_RIDERSHIP_RATE
    acad_mult = get_academic_multiplier(date_obj)
    N *= acad_mult
    
    weather_mult = 1.0
    if weather == 'rain': weather_mult = 1.0 if acad_mult > 1.3 else 0.9
    elif weather == 'snow': weather_mult = 0.6
    elif weather in ['sunny', 'clear']: weather_mult = 1.05
    
    is_weekend = date_obj.weekday() >= 5
    N *= 0.6 if is_weekend else 1.1
    return int(N * weather_mult)

# --- SIMULATION LOOP ---
def simulate_ridership(filename, start_date_str, num_days):
    df = load_and_prep_data(filename)
    stops_coords = df[['stop_lat', 'stop_lon']].values
    stop_ids = df['stop_id'].values
    
    start_date = datetime.datetime.strptime(start_date_str, "%d/%m/%Y")
    current_population = INITIAL_LONDON_POP
    all_trips = []
    
    print(f"Generating {num_days} days starting {start_date_str}...")
    
    for i in range(num_days):
        curr_date = start_date + datetime.timedelta(days=i)
        date_str = curr_date.strftime("%d/%m/%Y")
        wd = curr_date.weekday()
        
        #If today is Sept 1st, simulate student influx
        if curr_date.month == 9 and curr_date.day == 1 and i > 0:
            growth_rate = np.random.uniform(0.07, 0.10)
            increase = int(current_population * growth_rate)
            current_population += increase
            print(f"  [Sept 1] Population Growth: +{increase}. New Total: {current_population}")
        
        weather = np.random.choice(['rain', 'clear', 'snow'], p=[0.25, 0.65, 0.1])
        N_trips = calculate_daily_trips(curr_date, weather, current_population)
        
        #downtown Boost
        downtown_boost = 1.0
        if wd == 3: downtown_boost = 1.2
        elif wd == 4: downtown_boost = 1.5
        elif wd == 5: downtown_boost = 1.8
        elif wd == 6: downtown_boost = 1.4
        current_attr = df['base_attr'].copy()
        current_attr[df['type'] == 'Downtown'] *= downtown_boost
        
        #gravity Logic
        p_origins = df['base_pop'].values / df['base_pop'].sum()
        origin_indices = np.random.choice(len(df), size=N_trips, p=p_origins)
        unique_origins, counts = np.unique(origin_indices, return_counts=True)
        
        day_trips = []
        for o_idx, count in zip(unique_origins, counts):
            o_lat, o_lon = stops_coords[o_idx]
            dists = np.sqrt((stops_coords[:,0] - o_lat)**2 + (stops_coords[:,1] - o_lon)**2)
            dists = np.maximum(dists, 0.002)
            scores = current_attr.values / (dists ** 2.0)
            probs = scores / scores.sum()
            d_indices = np.random.choice(len(df), size=count, p=probs)
            
            for d_idx in d_indices:
                h = np.random.randint(6, 23)
                time_str = f"{h:02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}"
                day_trips.append([stop_ids[o_idx], stop_ids[d_idx], date_str, time_str, weather])
        
        all_trips.extend(day_trips)
        
    return pd.DataFrame(all_trips, columns=['Origin ID', 'Destination ID', 'Date', 'Time', 'Weather'])

# --- EXECUTION ---
final_df = simulate_ridership('stopwithZones.csv', "20/11/2025", 365)
final_df.to_csv("london_ridership_fixed_season.csv", index=False)
print("Done! File saved.")