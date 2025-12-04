#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from datetime import date, timedelta

from RIPR_B_predict import predict_total_riders  # Rider B

# ---------------- CONFIG ----------------

# First day that you have an .npy file for:
start_date = date(2021, 11, 18)   # <-- change if needed
num_days = 7                      # exactly 7 days

# Weather label passed to Rider B (same for all 7 days for now)
WEATHER_LABEL = "clear"

# Path pattern for Rider-A outputs
MLR_PATTERN = "mlr_output_{year}_{month:02d}_{day:02d}.npy"

# Stops file with real stop IDs
STOPS_FILE = "stops.txt"

# Only keep OD pairs with at least this many expected riders
MIN_RIDERS_EXPECTED = 0.01   # you can lower or raise this


# -------------- LOAD STOP IDS -------------

stops = pd.read_csv(STOPS_FILE)
stop_ids = stops["stop_id"].tolist()   # index 0 -> stop_ids[0], etc.
print(f"Loaded {len(stop_ids)} stops from {STOPS_FILE}")


# -------------- MAIN LOOP: 7 DAYS ---------

rows = []  # will hold all rows for all 7 days

for offset in range(num_days):
    current = start_date + timedelta(days=offset)
    y, m, d = current.year, current.month, current.day
    date_str = current.isoformat()

    # ---- Rider A: load shape matrix for this day ----
    fname = MLR_PATTERN.format(year=y, month=m, day=d)
    print(f"Processing {date_str} using {fname} ...")

    try:
        shape_matrix = np.load(fname)  # N x N
    except FileNotFoundError:
        print(f"  WARNING: file {fname} not found, skipping this day.")
        continue

    total_mass = shape_matrix.sum()
    if total_mass == 0:
        print(f"  WARNING: shape_matrix sum is 0 for {date_str}, skipping day.")
        continue

    # Trip Probability Matrix (Rider A normalized)
    tpm = shape_matrix / total_mass

    # ---- Rider B: total riders for this day ----
    total_riders = predict_total_riders(current, WEATHER_LABEL, temp=None)
    print(f"  Rider B total_riders = {total_riders}")

    # Expected riders per OD pair (decimals)
    riders_matrix = total_riders * tpm   # float, NOT rounded

    # Take all OD pairs with at least MIN_RIDERS_EXPECTED expected riders
    origin_idx, dest_idx = np.where(riders_matrix >= MIN_RIDERS_EXPECTED)

    for i, j in zip(origin_idx, dest_idx):
        rows.append({
            "date": date_str,
            "origin_stop": stop_ids[i],
            "dest_stop": stop_ids[j],
            "riders_expected": riders_matrix[i, j]
        })

# -------------- BUILD FINAL TABLE ----------

df_7days = pd.DataFrame(rows)

print("\nSample of 7-day RIPR expected-riders output:")
print(df_7days.head())

# Save everything (all 7 days) to CSV
df_7days.to_csv("ripr_7days_expected.csv", index=False)
print("\nSaved full 7-day output to ripr_7days_expected.csv")


# In[4]:


import pandas as pd
import numpy as np

df = pd.read_csv("ripr_7days_expected.csv")

# Poisson sample each value
rng = np.random.default_rng(0)  # reproducible results
df["riders_int"] = rng.poisson(df["riders_expected"])

df.to_csv("ripr_7days_int_poisson.csv", index=False)

print("Saved: ripr_7days_int_poisson.csv")
print(df.head())

