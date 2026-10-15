# ================================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# FEATURE ENGINEERING
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_PATH = "data/preprocessed/station_hourly_demand.csv"

OUTPUT_DIR = "data/features"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "station_hourly_demand_features.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("URBAN BIKE-SHARING DEMAND PREDICTION")
print("FEATURE ENGINEERING")
print("=" * 70)


# ================================================================
# 1. LOAD PREPROCESSED DATA
# ================================================================

print("\n1. LOADING PREPROCESSED DATA")
print("-" * 70)

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

required_columns = [
    "start_station_id",
    "start_station_name",
    "hour_timestamp",
    "demand"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are available.")


# ================================================================
# 2. CONVERT TIMESTAMP
# ================================================================

print("\n2. CONVERTING TIMESTAMP")
print("-" * 70)

df["hour_timestamp"] = pd.to_datetime(
    df["hour_timestamp"],
    errors="coerce"
)

invalid_timestamps = df["hour_timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps}")

if invalid_timestamps > 0:
    raise ValueError(
        "Invalid timestamps found."
    )

print(
    f"Start: {df['hour_timestamp'].min()}"
)
print(
    f"End  : {df['hour_timestamp'].max()}"
)


# ================================================================
# 3. SORT DATA CHRONOLOGICALLY
# ================================================================

print("\n3. SORTING DATA")
print("-" * 70)

df = df.sort_values(
    ["start_station_id", "hour_timestamp"]
).reset_index(drop=True)

print("Data sorted by station and timestamp.")


# ================================================================
# 4. CHECK STATION-HOUR DUPLICATES
# ================================================================

print("\n4. CHECKING DUPLICATES")
print("-" * 70)

duplicates = df.duplicated(
    subset=[
        "start_station_id",
        "hour_timestamp"
    ]
).sum()

print(
    f"Duplicate station-hour records: {duplicates}"
)

if duplicates > 0:
    raise ValueError(
        "Duplicate station-hour records detected."
    )


# ================================================================
# 5. CALENDAR FEATURES
# ================================================================

print("\n5. CREATING CALENDAR FEATURES")
print("-" * 70)

df["hour"] = df["hour_timestamp"].dt.hour
df["day"] = df["hour_timestamp"].dt.day
df["weekday"] = df["hour_timestamp"].dt.weekday
df["month"] = df["hour_timestamp"].dt.month
df["year"] = df["hour_timestamp"].dt.year
df["day_of_year"] = df["hour_timestamp"].dt.dayofyear
df["week_of_year"] = df["hour_timestamp"].dt.isocalendar().week.astype(int)

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

df["is_weekday"] = (
    df["weekday"] < 5
).astype(int)

# Calendar event encoding
EVENT_MAPPING = {
    "No Event": 0,
    "Christmas": 1,
    "New Year": 2,
    "Halloween": 3,
    "Concert": 4,
    "Music Festival": 5,
    "Food Festival": 6,
    "Street Festival": 7,
    "Marathon / Cycling Event": 8
}

if "special_event" in df.columns:
    df["special_event"] = df["special_event"].map(EVENT_MAPPING).fillna(0).astype(int)

print("Calendar features created.")


# ================================================================
# 6. PEAK-HOUR FEATURES
# ================================================================

print("\n6. CREATING PEAK-HOUR FEATURES")
print("-" * 70)

# Morning commute: 07:00–09:00 on weekdays
df["morning_peak"] = (
    (df["weekday"] < 5) &
    (df["hour"].between(7, 9))
).astype(int)

# Evening commute: 16:00–19:00 on weekdays
df["evening_peak"] = (
    (df["weekday"] < 5) &
    (df["hour"].between(16, 19))
).astype(int)

df["peak_hour"] = (
    (df["morning_peak"] == 1) |
    (df["evening_peak"] == 1)
).astype(int)

print("Peak-hour features created.")


# ================================================================
# 7. CYCLICAL TIME FEATURES
# ================================================================

print("\n7. CREATING CYCLICAL FEATURES")
print("-" * 70)

# Hour of day
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

# Day of week
df["weekday_sin"] = np.sin(
    2 * np.pi * df["weekday"] / 7
)

df["weekday_cos"] = np.cos(
    2 * np.pi * df["weekday"] / 7
)

# Month
df["month_sin"] = np.sin(
    2 * np.pi * (df["month"] - 1) / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * (df["month"] - 1) / 12
)

# Day of year
df["day_of_year_sin"] = np.sin(
    2 * np.pi * (df["day_of_year"] - 1) / 365.25
)

df["day_of_year_cos"] = np.cos(
    2 * np.pi * (df["day_of_year"] - 1) / 365.25
)

print("Cyclical features created.")


# ================================================================
# 8. STATION-HOUR INTERACTION FEATURES
# ================================================================

print("\n8. CREATING STATION-TIME INTERACTION FEATURES")
print("-" * 70)

df["hour_weekday"] = (
    df["hour"] * 7 + df["weekday"]
)

df["weekend_hour"] = (
    df["hour"] * df["is_weekend"]
)

df["weekday_hour"] = (
    df["hour"] * df["is_weekday"]
)

df["month_hour"] = (
    df["month"] * 24 + df["hour"]
)

print("Station-time interaction features created.")


# ================================================================
# 9. CREATE STATION-GROUPED TARGET SERIES
# ================================================================

print("\n9. PREPARING STATION-WISE DEMAND HISTORY")
print("-" * 70)

grouped_demand = df.groupby(
    "start_station_id"
)["demand"]


# ================================================================
# 10. LAG FEATURES
# ================================================================

print("\n10. CREATING LAG FEATURES")
print("-" * 70)

lag_periods = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168
]

for lag in lag_periods:

    df[f"lag_{lag}h"] = grouped_demand.shift(lag)

    print(
        f"Created lag_{lag}h"
    )


# ================================================================
# 11. SAME-HOUR HISTORICAL FEATURES
# ================================================================

print("\n11. CREATING SAME-HOUR HISTORICAL FEATURES")
print("-" * 70)

df["same_hour_previous_day"] = (
    grouped_demand.shift(24)
)

df["same_hour_previous_2days"] = (
    grouped_demand.shift(48)
)

df["same_hour_previous_3days"] = (
    grouped_demand.shift(72)
)

df["same_hour_previous_week"] = (
    grouped_demand.shift(168)
)

print("Same-hour historical features created.")


# ================================================================
# 12. DEMAND CHANGE FEATURES
# ================================================================

print("\n12. CREATING DEMAND CHANGE FEATURES")
print("-" * 70)

df["demand_change_1h"] = (
    df["demand"] -
    df["lag_1h"]
)

df["demand_change_24h"] = (
    df["demand"] -
    df["lag_24h"]
)

df["demand_change_week"] = (
    df["demand"] -
    df["lag_168h"]
)

print("Demand change features created.")


# ================================================================
# 13. ROLLING FEATURES
# ================================================================

print("\n13. CREATING LEAKAGE-FREE ROLLING FEATURES")
print("-" * 70)

# IMPORTANT:
# shift(1) ensures the current target is NEVER included
# in its own rolling statistics.

rolling_windows = [
    3,
    6,
    12,
    24,
    168
]

for window in rolling_windows:

    df[f"rolling_mean_{window}h"] = (
        df.groupby("start_station_id")["demand"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(window=window, min_periods=1)
             .mean()
        )
    )

    print(
        f"Created rolling_mean_{window}h"
    )


# ================================================================
# 14. ROLLING MAXIMUM
# ================================================================

print("\n14. CREATING ROLLING MAXIMUM FEATURES")
print("-" * 70)

df["rolling_max_24h"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(24, min_periods=1)
         .max()
    )
)

df["rolling_max_168h"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(168, min_periods=1)
         .max()
    )
)

print("Rolling maximum features created.")


# ================================================================
# 15. ROLLING MINIMUM
# ================================================================

print("\n15. CREATING ROLLING MINIMUM FEATURES")
print("-" * 70)

df["rolling_min_24h"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(24, min_periods=1)
         .min()
    )
)

df["rolling_min_168h"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(168, min_periods=1)
         .min()
    )
)

print("Rolling minimum features created.")


# ================================================================
# 16. ROLLING STANDARD DEVIATION
# ================================================================

print("\n16. CREATING ROLLING STANDARD DEVIATION")
print("-" * 70)

df["rolling_std_24h"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(24, min_periods=2)
         .std()
    )
)

df["rolling_std_168h"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(168, min_periods=2)
         .std()
    )
)

print("Rolling standard deviation features created.")


# ================================================================
# 17. ROLLING MEDIAN
# ================================================================

print("\n17. CREATING ROLLING MEDIAN")
print("-" * 70)

df["rolling_median_24h"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(24, min_periods=1)
         .median()
    )
)

print("Rolling median created.")


# ================================================================
# 18. HISTORICAL STATION DEMAND
# ================================================================

print("\n18. CREATING HISTORICAL STATION FEATURES")
print("-" * 70)

# Expanding statistics use only previous observations.
df["historical_demand"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .expanding(min_periods=1)
         .mean()
    )
)

df["historical_max_demand"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .expanding(min_periods=1)
         .max()
    )
)

df["historical_min_demand"] = (
    df.groupby("start_station_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .expanding(min_periods=1)
         .min()
    )
)

print("Historical station features created.")


# ================================================================
# 19. STATION-HOUR HISTORICAL AVERAGE
# ================================================================

print("\n19. CREATING STATION-HOUR HISTORICAL FEATURES")
print("-" * 70)

# These features are calculated using ONLY previous observations.
# The current demand is excluded.

df["station_hour_historical_mean"] = (
    df.groupby(
        ["start_station_id", "hour"]
    )["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .expanding(min_periods=1)
         .mean()
    )
)

df["station_weekday_hour_mean"] = (
    df.groupby(
        ["start_station_id", "weekday", "hour"]
    )["demand"]
    .transform(
        lambda x:
        x.shift(1)
         .expanding(min_periods=1)
         .mean()
    )
)

print("Station-hour historical features created.")


# ================================================================
# 20. DEMAND RATIO FEATURES
# ================================================================

print("\n20. CREATING DEMAND RATIO FEATURES")
print("-" * 70)

# Ratio of recent demand to longer-term demand.

df["lag_1h_ratio_24h"] = (
    df["lag_1h"] /
    (df["rolling_mean_24h"] + 1e-6)
)

df["lag_24h_ratio_week"] = (
    df["lag_24h"] /
    (df["rolling_mean_168h"] + 1e-6)
)

df["rolling_ratio_24h_168h"] = (
    df["rolling_mean_24h"] /
    (df["rolling_mean_168h"] + 1e-6)
)

print("Demand ratio features created.")


# ================================================================
# 21. CHECK FOR TARGET LEAKAGE
# ================================================================

print("\n21. TARGET LEAKAGE CHECK")
print("-" * 70)

feature_columns = [
    col for col in df.columns
    if col not in [
        "demand",
        "start_station_name",
        "hour_timestamp"
    ]
]

# Direct target duplication
direct_leakage = "demand" in feature_columns

print(
    f"Target demand inside feature list: {direct_leakage}"
)

# Check correlation between lag_1h and current demand
lag_check = (
    df["lag_1h"].notna()
)

if lag_check.sum() > 0:

    lag_matches = (
        df.loc[lag_check, "lag_1h"].values ==
        df.groupby("start_station_id")["demand"]
        .shift(1)
        .loc[lag_check]
        .values
    ).all()

    print(
        f"lag_1h correctly represents previous demand: "
        f"{lag_matches}"
    )


# ================================================================
# 22. MISSING VALUE CHECK
# ================================================================

print("\n22. MISSING VALUE CHECK")
print("-" * 70)

missing_before = df.isna().sum().sum()

print(
    f"Missing values before filling: {missing_before:,}"
)


# ================================================================
# 23. HANDLE INITIAL LAG/ROLLING VALUES
# ================================================================

print("\n23. HANDLING INITIAL HISTORICAL VALUES")
print("-" * 70)

# Initial observations of every station do not have enough
# historical information for long lag features.

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

df[numeric_columns] = (
    df[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)

# Fill historical feature gaps with 0.
# This only affects the beginning of each station's history.

df[numeric_columns] = (
    df[numeric_columns]
    .fillna(0)
)

missing_after = df.isna().sum().sum()

print(
    f"Missing values after processing: {missing_after:,}"
)


# ================================================================
# 24. FEATURE LIST
# ================================================================

print("\n24. FINAL FEATURE LIST")
print("-" * 70)

excluded_columns = [
    "demand",
    "start_station_name",
    "hour_timestamp"
]

model_features = [
    col for col in df.columns
    if col not in excluded_columns
]

print(
    f"Number of model features: {len(model_features)}"
)

for i, feature in enumerate(
    model_features,
    start=1
):
    print(
        f"{i:3}. {feature}"
    )


# ================================================================
# 25. FINAL DATASET VALIDATION
# ================================================================

print("\n25. FINAL DATASET VALIDATION")
print("-" * 70)

print(
    f"Rows              : {len(df):,}"
)

print(
    f"Columns           : {len(df.columns)}"
)

print(
    f"Model features    : {len(model_features)}"
)

print(
    f"Missing values    : {df.isna().sum().sum():,}"
)

print(
    f"Duplicate records : "
    f"{df.duplicated(subset=['start_station_id', 'hour_timestamp']).sum():,}"
)

print(
    f"Minimum demand    : {df['demand'].min()}"
)

print(
    f"Maximum demand    : {df['demand'].max()}"
)

print(
    f"Mean demand       : {df['demand'].mean():.4f}"
)

print(
    f"Std demand        : {df['demand'].std():.4f}"
)


# ================================================================
# 26. SAVE FEATURE-ENGINEERED DATASET
# ================================================================

print("\n26. SAVING FEATURE-ENGINEERED DATASET")
print("-" * 70)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "Feature-engineered dataset saved to:"
)

print(
    os.path.abspath(OUTPUT_PATH)
)


# ================================================================
# 27. SAVE FEATURE LIST
# ================================================================

feature_list_path = os.path.join(
    OUTPUT_DIR,
    "feature_list.csv"
)

feature_list_df = pd.DataFrame({
    "feature_number": range(
        1,
        len(model_features) + 1
    ),
    "feature": model_features
})

feature_list_df.to_csv(
    feature_list_path,
    index=False
)

print(
    "\nFeature list saved to:"
)

print(
    os.path.abspath(feature_list_path)
)


# ================================================================
# 28. FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print(
    f"Input rows                  : {len(df):,}"
)

print(
    f"Final columns               : {len(df.columns)}"
)

print(
    f"Model features              : {len(model_features)}"
)

print(
    f"Stations                    : "
    f"{df['start_station_id'].nunique()}"
)

print(
    f"Start timestamp             : "
    f"{df['hour_timestamp'].min()}"
)

print(
    f"End timestamp               : "
    f"{df['hour_timestamp'].max()}"
)

print(
    f"Target minimum              : "
    f"{df['demand'].min()}"
)

print(
    f"Target maximum              : "
    f"{df['demand'].max()}"
)

print(
    f"Target mean                 : "
    f"{df['demand'].mean():.4f}"
)

print(
    f"Missing values              : "
    f"{df.isna().sum().sum()}"
)

print(
    f"Duplicate station-hours     : "
    f"{df.duplicated(subset=['start_station_id', 'hour_timestamp']).sum()}"
)

print("\nOutput files:")

print(
    os.path.abspath(OUTPUT_PATH)
)

print(
    os.path.abspath(feature_list_path)
)

print("\n")
print("=" * 70)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNext step:")
print("Run model training with a chronological train/validation/test split.")