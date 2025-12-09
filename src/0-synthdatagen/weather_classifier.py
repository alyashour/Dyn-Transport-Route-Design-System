import pandas as pd
import numpy as np

# 1. Load the original daily data
df = pd.read_csv('in/london_weather_classified.csv')

# 2. Filter for dates starting from Jan 1, 2020
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)

# 3. Determine Weather Class (Same logic as before)
df['avg_temp'] = (df['max_temperature_v'] + df['min_temperature_v']) / 2
precip_threshold = df['precipitation_v'].median()

def classify_daily_weather(row):
    if row['precipitation_v'] < precip_threshold:
        return 'Clear'
    elif row['avg_temp'] <= 0:
        return 'Snowy'
    else:
        return 'Rainy'

df['Daily_Weather'] = df.apply(classify_daily_weather, axis=1)

# 4. Generate Hourly Data
hourly_rows = []

for _, row in df.iterrows():
    date = row['date']
    t_min = row['min_temperature_v']
    t_max = row['max_temperature_v']
    weather = row['Daily_Weather']
    
    # Calculate Average and Amplitude for the day
    avg = (t_max + t_min) / 2
    amp = (t_max - t_min) / 2
    
    for hour in range(24):
        # Estimate Temp: Min at 4 AM, Max at 4 PM (16:00)
        # Using a negative cosine wave shifted by 4 hours
        estimated_temp = avg + amp * -np.cos((hour - 4) * 2 * np.pi / 24)
        
        hourly_rows.append({
            'Date': date,
            'Hour': hour,
            'Datetime': date + pd.Timedelta(hours=hour),
            'Estimated_Temperature_C': round(estimated_temp, 2),
            'Weather_Condition': weather
        })

hourly_df = pd.DataFrame(hourly_rows)

# 5. Save to CSV
hourly_df.to_csv('dataset_generator/london_weather_hourly_estimated.csv', index=False)
print("Hourly estimation complete.")