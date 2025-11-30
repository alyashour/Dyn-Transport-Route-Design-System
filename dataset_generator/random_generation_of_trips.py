import pandas as pd
import numpy as np
import datetime

# --- CONFIGURATION ---
INITIAL_LONDON_POP = 422000  
BASE_RIDERSHIP_RATE = 0.12   

# Updated Weights to match LTC Data (55% of rides are students)
ZONE_WEIGHTS = {
    'Residential': {'pop': 100, 'attr': 10},
    'University':  {'pop': 300, 'attr': 1200}, 
    'Hub':         {'pop': 50,  'attr': 300},
    'Downtown':    {'pop': 50,  'attr': 500},
}

# --- DATA PREP ---
def load_and_prep_data(filename):
    df = pd.read_csv(filename)
    
    def get_stop_type(row):
        name = str(row['stop_name']).lower()
        zone = str(row['Name'])
        # Western & Fanshawe keywords
        if any(x in name for x in ['western', 'university', 'natural science', 'alumni', 'fanshawe', 'college']): return 'University'
        # Key Malls/Hubs
        if any(x in name for x in ['masonville', 'white oaks', 'argyle', 'westmount', 'cherryhill']): return 'Hub'
        # Downtown Core
        if zone == 'Central_London' or 'downtown' in name or 'richmond' in name: return 'Downtown'
        return 'Residential'

    df['type'] = df.apply(get_stop_type, axis=1)
    df['base_pop'] = df['type'].map(lambda x: ZONE_WEIGHTS.get(x, ZONE_WEIGHTS['Residential'])['pop'])
    df['base_attr'] = df['type'].map(lambda x: ZONE_WEIGHTS.get(x, ZONE_WEIGHTS['Residential'])['attr'])
    return df

# --- DYNAMIC FACTORS ---
def get_academic_multiplier(date_obj):
    year, m, d = date_obj.year, date_obj.month, date_obj.day
    current = datetime.datetime(year, m, d)
    
    #key academic dates for western/fanshawe
    f_exam_start = datetime.datetime(year, 12, 8)
    f_exam_end = datetime.datetime(year, 12, 22)
    w_exam_start = datetime.datetime(year, 4, 10)
    w_exam_end = datetime.datetime(year, 4, 30)

    #summer (May-Aug): SIGNIFICANT drop due to students leaving
    if 5 <= m <= 8: return 0.55 # Slightly lower than 0.6 to reflect student exodus
    
    #winter Break
    if m == 1 and d <= 7: return 0.6
    if m == 12 and d >= 23: return 0.6

    #fall/winter term dynamics
    delta_f = (f_exam_start - current).days
    if 0 < delta_f <= 25: return 1.3 + (0.3 * (1 - (delta_f / 25.0))) # Pre-exam crunch
    if f_exam_start <= current <= f_exam_end: return 1.3 - (0.7 * ((current - f_exam_start).days / (f_exam_end - f_exam_start).days))

    delta_w = (w_exam_start - current).days
    if 0 < delta_w <= 25: return 1.3 + (0.3 * (1 - (delta_w / 25.0)))
    if w_exam_start <= current <= w_exam_end: return 1.3 - (0.7 * ((current - w_exam_start).days / (w_exam_end - w_exam_start).days))
        
    return 1.3 #standard Term Time

def calculate_base_demand(date_obj, current_pop):
    N = current_pop * BASE_RIDERSHIP_RATE
    acad_mult = get_academic_multiplier(date_obj)
    N *= acad_mult
    
    wd = date_obj.weekday()
    is_weekend = wd >= 5
    
    #weekend reduction
    if is_weekend:
        N *= 0.6
        #sunday Service Reduction: Sunday schedules are lighter than Saturday
        if wd == 6: 
            N *= 0.75 
    else:
        N *= 1.1
        
    return int(N)

def apply_weather_rules(temps, conditions, month):
    probs = np.ones_like(temps, dtype=float)
    
    probs[temps <= -20] = 0.10  # Extreme Cold Warning
    mask_20_10 = (temps > -20) & (temps <= -10)
    probs[mask_20_10] = 0.30
    mask_10_0 = (temps > -10) & (temps <= 0)
    probs[mask_10_0] = 0.90 # Londoners are used to -5, minimal impact
    
    #summer Heat
    if 5 <= month <= 8:
        probs[temps > 30] = 0.30
        mask_25_30 = (temps > 25) & (temps <= 30)
        probs[mask_25_30] = 0.70
    
    #conditions impact
    probs[conditions == 'Rainy'] *= 0.75
    probs[conditions == 'Snowy'] *= 0.60
        
    return probs

# --- SIMULATION LOOP ---
def simulate_ridership(filename, start_date_str, num_days, weather_file='dataset_generator/london_weather_hourly_estimated.csv'):
    df = load_and_prep_data(filename)
    stops_coords = df[['stop_lat', 'stop_lon']].values
    stop_ids = df['stop_id'].values
    stop_types = df['type'].values
    
    print(f"Loading weather data from {weather_file}...")
    try:
        w_df = pd.read_csv(weather_file)
        w_df['Date'] = pd.to_datetime(w_df['Date'])
        weather_lookup = w_df.groupby(w_df['Date'].dt.date)['Estimated_Temperature_C'].apply(np.array).to_dict()
        cond_lookup = w_df.groupby(w_df['Date'].dt.date)['Weather_Condition'].apply(np.array).to_dict()
    except Exception as e:
        print(f"Error loading weather file: {e}")
        return None

    start_date = datetime.datetime.strptime(start_date_str, "%d/%m/%Y")
    current_population = INITIAL_LONDON_POP
    all_trips = []
    
    print(f"Generating {num_days} days starting {start_date_str}...")
    
    for i in range(num_days):
        curr_date = start_date + datetime.timedelta(days=i)
        date_str = curr_date.strftime("%d/%m/%Y")

        c_day = curr_date.day
        c_month = curr_date.month
        c_year = curr_date.year

        wd = curr_date.weekday()
        
        # --- NIGHTLIFE LOGIC ---
        #thursday (Student Nights), Friday, Saturday (Peak)
        if wd == 3:   #thursday
            downtown_boost = 1.3
            late_return_prob = 0.60
        elif wd == 4: #friday
            downtown_boost = 1.5
            late_return_prob = 0.80
        elif wd == 5: #saturday
            downtown_boost = 1.9
            late_return_prob = 0.95 
        elif wd == 6: #sunday
            downtown_boost = 1.1
            late_return_prob = 0.20
        else:
            downtown_boost = 1.0
            late_return_prob = 0.0

        is_nightlife_day = (wd in [3, 4, 5, 6])

        #population Growth (Sept 1) - Massive Student Influx
        if curr_date.month == 9 and curr_date.day == 1 and i > 0:
            growth_pct = np.random.uniform(0.05, 0.07) 
            increase = int(current_population * growth_pct)
            current_population += increase
            print(f"  [Sept 1] Student Influx: +{increase}. New Total: {current_population}")
        
        # 1.calculate Potential Demand
        total_potential_trips = calculate_base_demand(curr_date, current_population)
        num_people = total_potential_trips // 2
        
        # 2. Get Weather (Temperature AND Conditions)
        lookup_date = curr_date.date()
        if lookup_date not in weather_lookup:
            daily_temps = np.full(24, 20.0)
            daily_conds = np.full(24, 'Clear', dtype=object)
        else:
            daily_temps = weather_lookup[lookup_date]
            daily_conds = cond_lookup[lookup_date]

            # Fix missing hours (Padding)
            if len(daily_temps) < 24: 
                pad_len = 24 - len(daily_temps)
                daily_temps = np.pad(daily_temps, (0, pad_len), 'edge')
                daily_conds = np.pad(daily_conds, (0, pad_len), constant_values='Clear')
            elif len(daily_temps) > 24: 
                daily_temps = daily_temps[:24]
                daily_conds = daily_conds[:24]

        # 3.generate Times
        start_weights = np.array([0]*6 + [0.08, 0.12, 0.10, 0.07, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.10, 0.08, 0.05, 0.02])
        #add zeros to reach 24 hours if needed (len is currently 20)
        if len(start_weights) < 24:
            start_weights = np.pad(start_weights, (0, 24-len(start_weights)), 'constant')
        start_weights = start_weights / start_weights.sum()
        
        start_hours = np.random.choice(np.arange(24), size=num_people, p=start_weights)
        
        durations = np.random.normal(loc=5, scale=2.5, size=num_people).astype(int)
        durations = np.clip(durations, 1, 10)
        return_hours = start_hours + durations
        
        #weather Check (Start Time)
        start_temps = daily_temps[start_hours]
        start_conds = daily_conds[start_hours] 
        weather_probs = apply_weather_rules(start_temps, start_conds, curr_date.month)
        
        random_checks = np.random.random(size=num_people)
        weather_accept_mask = random_checks < weather_probs
        
        accepted_starts = start_hours[weather_accept_mask]
        accepted_returns = return_hours[weather_accept_mask]
        N_actual_pairs = len(accepted_starts)
        
        if N_actual_pairs > 0:
            #downtown Boost
            current_attr = df['base_attr'].copy()
            current_attr[df['type'] == 'Downtown'] *= downtown_boost
            
            p_origins = df['base_pop'].values / df['base_pop'].sum()
            origin_indices = np.random.choice(len(df), size=N_actual_pairs, p=p_origins)
            unique_origins, counts = np.unique(origin_indices, return_counts=True)
            
            indices = np.arange(N_actual_pairs)
            np.random.shuffle(indices)
            shuffled_starts = accepted_starts[indices]
            shuffled_returns = accepted_returns[indices]
            
            cursor = 0
            day_trips = []
            
            for o_idx, count in zip(unique_origins, counts):
                o_lat, o_lon = stops_coords[o_idx]
                dists = np.sqrt((stops_coords[:,0] - o_lat)**2 + (stops_coords[:,1] - o_lon)**2)
                dists = np.maximum(dists, 0.002)
                scores = current_attr.values / (dists ** 2.0)
                probs = scores / scores.sum()
                
                d_indices = np.random.choice(len(df), size=count, p=probs)
                
                batch_starts = shuffled_starts[cursor : cursor+count]
                batch_returns = shuffled_returns[cursor : cursor+count]
                cursor += count
                
                for d_idx, t_out, t_ret in zip(d_indices, batch_starts, batch_returns):
                    dest_type = stop_types[d_idx]
                    final_ret_h = t_ret
                    final_ret_m = np.random.randint(0, 60)
                    record_return = True
                    
                    # --- NIGHTLIFE LOGIC ---
                    if is_nightlife_day and dest_type == 'Downtown':
                        if np.random.random() < late_return_prob: 
                            if np.random.random() < 0.4: 
                                final_ret_h = 22
                                final_ret_m = np.random.randint(30, 60)
                            else: 
                                final_ret_h = 23
                                final_ret_m = np.random.randint(0, 60)
                            
                            # Missed Bus / Taxi Home (30%)
                            if np.random.random() < 0.30:
                                record_return = False
                    
                    if record_return and final_ret_h >= 24:
                        record_return = False
                        
                    # 1. OUTBOUND
                    time_str_1 = f"{t_out:02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}"
                    day_trips.append([stop_ids[o_idx], stop_ids[d_idx], c_day, c_month, c_year])
                    
                    # 2. RETURN
                    if record_return:
                        time_str_2 = f"{final_ret_h:02d}:{final_ret_m:02d}:{np.random.randint(0,60):02d}"
                        day_trips.append([stop_ids[d_idx], stop_ids[o_idx], c_day, c_month, c_year])
            
            all_trips.extend(day_trips)
            
    return pd.DataFrame(all_trips, columns=['Origin ID', 'Destination ID', 'Day', 'Month', 'Year'])

# --- EXECUTION ---
final_df = simulate_ridership('dataset_generator/stopwithZones.csv', "18/11/2021", 1460, weather_file='dataset_generator/london_weather_hourly_estimated.csv')
final_df.to_csv("dataset_generator/london_ridership_trip_generation.csv", index=False)
print("Done!")
print(final_df.head(10))