import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


print("Loading trip data...")
df = pd.read_csv("data/london_ridership_trip_generation.csv")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

print("Aggregating daily ridership...")
daily = df.groupby("Date").size().reset_index(name="Daily_Ridership")

print("Loading weather data...")
weather = pd.read_csv("data/london_weather_classified.csv")
weather["Date"] = pd.to_datetime(weather["Date"])

print("Merging...")
full = daily.merge(weather, on="Date", how="left")

print("Feature engineering...")
full["DayOfWeek"] = full["Date"].dt.weekday
full["Month"] = full["Date"].dt.month
full["Day"] = full["Date"].dt.day

full["Weather_Label"] = full["Weather"].str.lower().map({
    "clear": 0, "sunny": 0,
    "rain": 1, "rainy": 1,
    "snow": 2, "snowy": 2,
}).fillna(0).astype(int)

full["Is_Exam_Season"] = full["Month"].isin([4, 12]).astype(int)

feature_cols = [
    "DayOfWeek", "Month", "Day",
    "Weather_Label",
    "Is_Exam_Season"
]

print("Using features:", feature_cols)

X = full[feature_cols]
y = full["Daily_Ridership"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nEvaluation:")
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"RMSE: {rmse:.2f}")
print("R²:", r2_score(y_test, y_pred))

print("\nSaving model...")
with open("RIPR_B/rider_count_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("RIPR-B Training Complete!")
