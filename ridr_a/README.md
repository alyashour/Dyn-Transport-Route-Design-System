# RIDR-A

## Training

### Data & Preprocessing

Ensure the following files are in the data directory: `city_population.csv, stop_data.csv, trips_data.csv, weather_hourly.csv`
With the following Schema (respectively):
- `Year,Population`
- `stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,stop_timezone,wheelchair_boarding`
- `Origin ID,Destination ID,Day,Month,Year`
- `Date,Hour,Datetime,Estimated_Temperature_C,Weather_Condition`

Then run the following scripts:
1. `summarize_weather_data.py data/weather_hourly.csv weahter.csv`. This summarizes the hourly data into daily highs and lows.
2. `consolidate_data.py`. This combines all the tables into a single table called "processed_combined_data.csv". Every row is 1 trip.
3. `group_daily_rows_data.py`. This groups every day into a single row in "daily_grouped_data.csv". Every row is 1 day.

### Training

Run `train.py`. This will produce a model in "models" called "mlr_grid_model_latest.pkl".

## Auxiliary Scripts

- `shrink_csv.py <input.csv> <output.csv> <num_rows>` copies the first `num_rows` of input.csv into output.csv. This is to create smaller datasets to use. E.g., data/trips_data_extra_small.csv and data/trips_data_small.csv.
- `visualize_model.py /path/to/model.pkl` creates images that visualize the proper distribution & the model's predicted distribution for any given day.
![alt text](images/61m-model/model_grid_2021-11-18.png)
![alt text](images/61m-model/true_grid_2021-11-18.png)