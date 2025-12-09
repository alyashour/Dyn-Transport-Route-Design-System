# Neural Network Models for Ridership Predictor

There are two flavours: shotgun and sniper.
Shotgun predicts many routes at the same time, sniper predicts only one at a time.

## Usage
- put `city_population.csv`, `stops.csv`, `trips.csv`, and `weather.csv` into the `/data/1_ripr/in` directory, 
- run `preprocess.py`, this creates a dataset.h5 file in `memcache/`,
- then run `shotgun_train.py`, 
- and finally `shotgun_demo.py`
