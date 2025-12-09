import csv
from collections import defaultdict, Counter
import sys

def convert_csv(in_file, out_file):
    data = defaultdict(lambda: {
        "temps": [],
        "conditions": []
    })

    # --- Read the original CSV ---
    with open(in_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["Date"]
            temp = float(row["Estimated_Temperature_C"])
            cond = row["Weather_Condition"]

            data[date]["temps"].append(temp)
            data[date]["conditions"].append(cond)

    # --- Write the new CSV ---
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "temp_high_c", "temp_low_c", "Weather_Condition"])

        for date, vals in sorted(data.items()):
            temps = vals["temps"]
            conditions = vals["conditions"]

            high = max(temps)
            low = min(temps)
            most_common_condition = Counter(conditions).most_common(1)[0][0]

            writer.writerow([date, f"{high:.2f}", f"{low:.2f}", most_common_condition])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_weather.py input.csv output.csv")
        sys.exit(1)

    convert_csv(sys.argv[1], sys.argv[2])
