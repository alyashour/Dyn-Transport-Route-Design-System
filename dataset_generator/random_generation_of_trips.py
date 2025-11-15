import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np 

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=input-files=++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("Loading static input files")
try:
    zones_df = pd.read_excel("zones.xlsx")
    demo_df = pd.read_excel("demographics.xlsx")
except FileNotFoundError:
    print("Error: zones.xlsx or demographics.xlsx not found.")
    exit()

#get lists of zone IDs, their populations, and their attractiveness
zone_ids = zones_df["Zone_ID"].tolist()
zone_populations = zones_df["Base_Population"].tolist()
zone_attractiveness = zones_df["Base_attractivness"].tolist()

#normalizing weights so they add up to 1
pop_weights = np.array(zone_populations) / sum(zone_populations)
attr_weights = np.array(zone_attractiveness) / sum(zone_attractiveness)

print("Checkpoint1: Static files loaded.")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=helper-functions=+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_season(date_obj): #returns the season for a given date.
    month = date_obj.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

def get_daily_weather(season): #generates a simple weather forecast based on the season.
    if season == "winter":
        return random.choices(["clear", "cloudy", "snow"], [0.4, 0.3, 0.3], k=1)[0]
    elif season == "summer":
        return random.choices(["clear", "cloudy", "rain"], [0.7, 0.2, 0.1], k=1)[0]
    else: #spring/fall
        return random.choices(["clear", "cloudy", "rain"], [0.5, 0.3, 0.2], k=1)[0]

def get_total_trips(day_of_week, weather, season): #calculates the total number of trips for the day based on context
    
    base_trips = 20000 #starting with a base number of trips for an average weekday
    
    # 1.adjust for the day of the week
    if day_of_week == 5: # saturday
        base_trips *= 0.7
    elif day_of_week == 6: # sunday
        base_trips *= 0.5
        
    # 2.adjust for weather
    if weather == "snow" or weather == "rain":
        base_trips *= 0.9  # 10% less riders in bad weather
    elif weather == "clear" and season == "summer":
        base_trips *= 1.1  # 10% more riders in nice weather
        
    # 3.adjust for school season (e.g., Western/Fanshawe)
    if season != "summer":
        base_trips *= 1.2 # 20% more riders when school is in
        
    return int(base_trips)

def generate_trip_time(day_of_week): #generates a random time for a trip, weighted by time of day.
    #define time "peaks" (hour, weight)
    if day_of_week < 5: # weekday
        #bimodal peaks: AM (7-9) and PM (16-18)
        peak_hours = [7, 8, 9, 15, 16, 17, 18]
        weights = [0.1, 0.15, 0.1, 0.08, 0.12, 0.12, 0.1]
    else: #weekend
        #unimodal peak: afternoon (12-16)
        peak_hours = [11, 12, 13, 14, 15, 16]
        weights = [0.1, 0.15, 0.15, 0.15, 0.1, 0.1]
        
    #normalizing weights
    weights = np.array(weights) / sum(weights)
    
    #choosing an hour based on the weights
    hour = np.random.choice(peak_hours, p=weights)
    
    #added random minutes/seconds to make it look real
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    #used am/pm format
    time_obj = datetime(2025, 1, 1, hour, minute, second)
    return time_obj.strftime("%I:%M:%S %p") # e.g., "08:09:22 AM"

def generate_trip_origin_dest(time_str): #generates a weighted origin and destination based on time
    
    #for this model, we'll just use the main population/attraction weights
    #NOTE to self: next iteration create a model where weights change based on time (e.g., AM peak)
    
    origin_id = np.random.choice(zone_ids, p=pop_weights)
    dest_id = np.random.choice(zone_ids, p=attr_weights)
    
    #ensuring origin and destination are not the same
    while origin_id == dest_id:
        dest_id = np.random.choice(zone_ids, p=attr_weights)
        
    return origin_id, dest_id

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++=main-Loop=+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# define the date range for simulation
start_date = datetime(2024, 11, 1)  #we can change this as needed
end_date = datetime(2025, 10, 31)
current_date = start_date
trip_counter = 1

# open the output files
with open("trip_data.xlsx", "w") as trip_file, \
     open("weather_data.xlsx", "w") as weather_file:
     
    #write headers
    trip_file.write("Trip_ID,Origin_ID,Destination_ID,Date,Time\n")
    weather_file.write("Date,Weather_Condition\n")
    
    #loop for each day
    while current_date <= end_date:
        date_str = current_date.strftime("%d/%m/%Y")
        day_of_week = current_date.weekday() # 0=Monday, 6=Sunday
        season = get_season(current_date)
        
        # 1.get daily context
        weather = get_daily_weather(season)
        total_trips = get_total_trips(day_of_week, weather, season)
        
        # 2.write this day's weather to the weather file
        weather_file.write(f"{date_str},{weather}\n")
        
        if day_of_week in [0,1,2,3,4]: # Print progress for Mondays-Fridays
           print(f"Generating {total_trips} trips for {date_str} (Weekday, {weather})...")
        
        # 3.generate all trips for this day
        for _ in range(total_trips):
            #generate time, origin, and destination
            trip_time = generate_trip_time(day_of_week)
            origin, dest = generate_trip_origin_dest(trip_time)
            
            #write the trip to the trip file
            trip_file.write(f"{trip_counter},{origin},{dest},{date_str},{trip_time}\n")
            trip_counter += 1
            
        #move to the next day
        current_date += timedelta(days=1)

print("checkpoint: generation complete!")
print(f"total trips generated: {trip_counter - 1}")
print("files created: trip_data.xlsx, weather_data.xlsx")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=The-End=+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++