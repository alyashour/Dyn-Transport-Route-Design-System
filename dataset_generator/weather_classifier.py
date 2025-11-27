import pandas as pd

# 1. Load the dataset
df = pd.read_csv('dataset_generator/weatherstats_london_normal_daily-2.csv')

# 2. Filter for dates starting from Jan 1, 2020
df['date'] = pd.to_datetime(df['date'])
df_filtered = df[df['date'] >= '2020-01-01'].copy().sort_values('date')

# 3. Define Logic
# Calculate average temp
df_filtered['avg_temp'] = (df_filtered['max_temperature_v'] + df_filtered['min_temperature_v']) / 2
# Determine median precipitation for the relative "Clear" threshold
precip_threshold = df_filtered['precipitation_v'].median()

def classify_weather(row):
    if row['precipitation_v'] < precip_threshold:
        return 'Clear'
    elif row['avg_temp'] <= 0:
        return 'Snowy'
    else:
        return 'Rainy'

# 4. Apply Classification
df_filtered['Weather'] = df_filtered.apply(classify_weather, axis=1)

# 5. Export specific columns to CSV
output_df = df_filtered[['date', 'Weather']]
output_df.columns = ['Date', 'Weather'] # Rename for clarity
output_df.to_csv('dataset_generator/london_weather_classified.csv', index=False)

print("File 'dataset_generator/london_weather_classified.csv' created successfully.")