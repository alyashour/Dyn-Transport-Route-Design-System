import pickle
import pandas as pd

with open("RIPR_B/rider_count_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_total_riders(date, weather_label, temp):
    """
    date: datetime.date or string
    weather_label: "clear" | "rain" | "snow"
    temp: temperature in C (optional, not used currently)
    """
    date = pd.to_datetime(date)

    features = {
        "DayOfWeek": date.weekday(),
        "Month": date.month,
        "Day": date.day,
        "Weather_Label": {"clear":0, "sunny":0, "rain":1, "snow":2}.get(weather_label.lower(), 0),
        "Is_Exam_Season": int(date.month in [4, 12])
    }

    X = pd.DataFrame([features])
    
    # Ensure column order matches training
    expected_cols = ["DayOfWeek", "Month", "Day", "Weather_Label", "Is_Exam_Season"]
    X = X[expected_cols]
    
    pred = model.predict(X)[0]
    return max(0, int(pred))  # Ensure non-negative

