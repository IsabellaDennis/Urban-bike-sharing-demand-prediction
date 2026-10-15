# ================================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# PREPROCESSING
# Dataset: trip.csv ONLY
# ================================================================

import os
import pandas as pd
import numpy as np


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "trip.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "preprocessed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "station_hourly_demand.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("URBAN BIKE-SHARING DEMAND PREDICTION")
print("PREPROCESSING")
print("=" * 70)


# ================================================================
# 2. LOAD RAW DATASET
# ================================================================

print("\n1. LOADING RAW DATASET")
print("-" * 70)

if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(
        f"trip.csv not found at:\n{RAW_FILE}"
    )

df = pd.read_csv(RAW_FILE)

print(f"Dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nOriginal columns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2d}. {col}")


# ================================================================
# 3. PRESERVE ORIGINAL DATASET INFORMATION
# ================================================================

original_rows = len(df)
original_columns = len(df.columns)


# ================================================================
# 4. CHECK REQUIRED COLUMNS
# ================================================================

print("\n2. CHECKING REQUIRED COLUMNS")
print("-" * 70)

required_columns = [
    "start_date",
    "start_station_name",
    "start_station_id"
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Required columns missing: {missing_required}"
    )

print("All required columns are available.")


# ================================================================
# 5. CHECK DUPLICATES
# ================================================================

print("\n3. CHECKING DUPLICATES")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count:,}")

if duplicate_count > 0:
    print("Removing duplicate rows...")
    df = df.drop_duplicates().copy()
else:
    print("No duplicate rows found.")


# ================================================================
# 6. CONVERT START DATE
# ================================================================

print("\n4. CONVERTING START DATE")
print("-" * 70)

df["start_date_parsed"] = pd.to_datetime(
    df["start_date"],
    errors="coerce",
    format="mixed"
)

invalid_dates = df["start_date_parsed"].isna().sum()

print(f"Invalid start dates: {invalid_dates:,}")

if invalid_dates > 0:
    print("Removing records with invalid start dates...")
    df = df.dropna(
        subset=["start_date_parsed"]
    ).copy()
else:
    print("All start dates parsed successfully.")


# ================================================================
# 7. CHECK STATION VALUES
# ================================================================

print("\n5. CHECKING STATION INFORMATION")
print("-" * 70)

missing_station_id = df["start_station_id"].isna().sum()
missing_station_name = df["start_station_name"].isna().sum()

print(f"Missing start station IDs   : {missing_station_id:,}")
print(f"Missing start station names : {missing_station_name:,}")

if missing_station_id > 0:
    df = df.dropna(
        subset=["start_station_id"]
    ).copy()

if missing_station_name > 0:
    df = df.dropna(
        subset=["start_station_name"]
    ).copy()

print(
    f"Unique start stations: "
    f"{df['start_station_id'].nunique():,}"
)


# ================================================================
# 8. STANDARDIZE STATION ID
# ================================================================

print("\n6. STANDARDIZING STATION IDs")
print("-" * 70)

df["start_station_id"] = pd.to_numeric(
    df["start_station_id"],
    errors="coerce"
)

df = df.dropna(
    subset=["start_station_id"]
).copy()

df["start_station_id"] = (
    df["start_station_id"]
    .astype(int)
)

print("Station IDs converted to integer.")


# ================================================================
# 9. CREATE HOURLY TIMESTAMP
# ================================================================

print("\n7. CREATING HOURLY TIMESTAMP")
print("-" * 70)

df["hour_timestamp"] = (
    df["start_date_parsed"]
    .dt.floor("h")
)

print(
    "Hourly timestamp created using "
    "start_date.floor('h')."
)

print(
    f"Start hour: "
    f"{df['hour_timestamp'].min()}"
)

print(
    f"End hour  : "
    f"{df['hour_timestamp'].max()}"
)


# ================================================================
# 10. CHECK STATION NAME CONSISTENCY
# ================================================================

print("\n8. CHECKING STATION NAME CONSISTENCY")
print("-" * 70)

station_mapping = (
    df[
        [
            "start_station_id",
            "start_station_name"
        ]
    ]
    .drop_duplicates()
    .sort_values("start_station_id")
)

station_name_counts = (
    station_mapping
    .groupby("start_station_id")
    ["start_station_name"]
    .nunique()
)

inconsistent_stations = (
    station_name_counts[
        station_name_counts > 1
    ]
)

print(
    f"Stations with multiple names: "
    f"{len(inconsistent_stations)}"
)

if len(inconsistent_stations) > 0:

    print(
        "Using the most frequently occurring "
        "station name for each station ID."
    )

    station_mapping = (
        df.groupby(
            "start_station_id"
        )["start_station_name"]
        .agg(
            lambda x: x.value_counts().index[0]
        )
        .reset_index()
    )

else:

    station_mapping = (
        station_mapping
        .drop_duplicates(
            subset=["start_station_id"]
        )
    )


# ================================================================
# 11. CREATE HOURLY DEMAND
# ================================================================

print("\n9. CALCULATING HOURLY STATION DEMAND")
print("-" * 70)

print(
    "Demand definition:"
)

print(
    "Number of trips departing from each "
    "station during each hour."
)

hourly_demand = (
    df.groupby(
        [
            "start_station_id",
            "hour_timestamp"
        ]
    )
    .size()
    .reset_index(
        name="demand"
    )
)

print(
    f"Observed station-hour records: "
    f"{len(hourly_demand):,}"
)

print(
    f"Positive-demand records: "
    f"{(hourly_demand['demand'] > 0).sum():,}"
)


# ================================================================
# 12. CREATE COMPLETE STATION-HOUR GRID
# ================================================================

print("\n10. CREATING COMPLETE STATION-HOUR GRID")
print("-" * 70)

stations = (
    df["start_station_id"]
    .drop_duplicates()
    .sort_values()
    .to_numpy()
)

start_hour = (
    df["hour_timestamp"].min()
)

end_hour = (
    df["hour_timestamp"].max()
)

all_hours = pd.date_range(
    start=start_hour,
    end=end_hour,
    freq="h"
)

print(f"Number of stations : {len(stations):,}")
print(f"Number of hours    : {len(all_hours):,}")

expected_records = (
    len(stations) *
    len(all_hours)
)

print(
    f"Expected station-hour records: "
    f"{expected_records:,}"
)


# ================================================================
# 13. CARTESIAN PRODUCT
# ================================================================

print("\n11. GENERATING COMPLETE TIMELINE")
print("-" * 70)

station_hour_grid = pd.MultiIndex.from_product(
    [
        stations,
        all_hours
    ],
    names=[
        "start_station_id",
        "hour_timestamp"
    ]
).to_frame(
    index=False
)

print(
    f"Complete grid created: "
    f"{len(station_hour_grid):,} rows"
)


# ================================================================
# 14. MERGE OBSERVED DEMAND
# ================================================================

print("\n12. MERGING OBSERVED DEMAND")
print("-" * 70)

processed = station_hour_grid.merge(
    hourly_demand,
    on=[
        "start_station_id",
        "hour_timestamp"
    ],
    how="left"
)

# Missing station-hours represent zero departures.
processed["demand"] = (
    processed["demand"]
    .fillna(0)
    .astype(int)
)

print(
    "Missing station-hour demand values "
    "converted to zero."
)


# ================================================================
# 15. ADD STATION NAMES
# ================================================================

print("\n13. ADDING STATION NAMES")
print("-" * 70)

processed = processed.merge(
    station_mapping,
    on="start_station_id",
    how="left"
)

# Map calendar dates to event categories
if "special_event" in df.columns:
    df["date_key"] = df["start_date_parsed"].dt.date
    event_map = df.groupby("date_key")["special_event"].first().to_dict()
    date_series = pd.to_datetime(processed["hour_timestamp"]).dt.date
    processed["special_event"] = date_series.map(event_map).fillna("No Event")

    # Reorder columns
    processed = processed[
        [
            "start_station_id",
            "start_station_name",
            "hour_timestamp",
            "demand",
            "special_event"
        ]
    ]
else:
    # Reorder columns
    processed = processed[
        [
            "start_station_id",
            "start_station_name",
            "hour_timestamp",
            "demand"
        ]
    ]


# ================================================================
# 16. SORT CHRONOLOGICALLY
# ================================================================

print("\n14. SORTING DATA")
print("-" * 70)

processed = processed.sort_values(
    [
        "start_station_id",
        "hour_timestamp"
    ]
).reset_index(
    drop=True
)

print(
    "Data sorted by station and timestamp."
)


# ================================================================
# 17. CHECK FOR DUPLICATE STATION-HOUR RECORDS
# ================================================================

print("\n15. CHECKING STATION-HOUR DUPLICATES")
print("-" * 70)

station_hour_duplicates = (
    processed
    .duplicated(
        subset=[
            "start_station_id",
            "hour_timestamp"
        ]
    )
    .sum()
)

print(
    f"Duplicate station-hour records: "
    f"{station_hour_duplicates:,}"
)

if station_hour_duplicates != 0:
    raise ValueError(
        "Duplicate station-hour records detected."
    )


# ================================================================
# 18. DEMAND QUALITY ANALYSIS
# ================================================================

print("\n16. DEMAND QUALITY ANALYSIS")
print("-" * 70)

total_records = len(processed)

zero_demand = (
    processed["demand"] == 0
).sum()

positive_demand = (
    processed["demand"] > 0
).sum()

negative_demand = (
    processed["demand"] < 0
).sum()

print(
    f"Total station-hour records : "
    f"{total_records:,}"
)

print(
    f"Zero-demand records        : "
    f"{zero_demand:,}"
)

print(
    f"Positive-demand records    : "
    f"{positive_demand:,}"
)

print(
    f"Negative-demand records    : "
    f"{negative_demand:,}"
)

print(
    f"Zero-demand percentage     : "
    f"{(zero_demand / total_records) * 100:.2f}%"
)

print(
    f"Positive-demand percentage : "
    f"{(positive_demand / total_records) * 100:.2f}%"
)

print(
    f"Minimum demand             : "
    f"{processed['demand'].min()}"
)

print(
    f"Maximum demand             : "
    f"{processed['demand'].max()}"
)

print(
    f"Mean demand                : "
    f"{processed['demand'].mean():.4f}"
)

print(
    f"Standard deviation         : "
    f"{processed['demand'].std():.4f}"
)


# ================================================================
# 19. CHECK MISSING VALUES
# ================================================================

print("\n17. FINAL MISSING VALUE CHECK")
print("-" * 70)

missing_summary = (
    processed
    .isnull()
    .sum()
)

total_missing = (
    missing_summary.sum()
)

print(
    f"Total missing values: "
    f"{total_missing:,}"
)

if total_missing > 0:

    print("\nMissing values by column:")
    print(
        missing_summary[
            missing_summary > 0
        ]
    )

else:

    print(
        "No missing values found."
    )


# ================================================================
# 20. CHECK DATA TYPES
# ================================================================

print("\n18. FINAL DATA TYPES")
print("-" * 70)

print(
    processed.dtypes
)


# ================================================================
# 21. TIMESTAMP CHECK
# ================================================================

print("\n19. TIMESTAMP VALIDATION")
print("-" * 70)

print(
    f"Start timestamp : "
    f"{processed['hour_timestamp'].min()}"
)

print(
    f"End timestamp   : "
    f"{processed['hour_timestamp'].max()}"
)

print(
    f"Unique timestamps: "
    f"{processed['hour_timestamp'].nunique():,}"
)


# ================================================================
# 22. STATION CHECK
# ================================================================

print("\n20. FINAL STATION CHECK")
print("-" * 70)

print(
    f"Unique stations: "
    f"{processed['start_station_id'].nunique():,}"
)

print(
    f"Unique station names: "
    f"{processed['start_station_name'].nunique():,}"
)


# ================================================================
# 23. SAVE PREPROCESSED DATASET
# ================================================================

print("\n21. SAVING PREPROCESSED DATASET")
print("-" * 70)

processed.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Preprocessed dataset saved to:"
)

print(
    OUTPUT_FILE
)


# ================================================================
# 24. VERIFY SAVED FILE
# ================================================================

print("\n22. VERIFYING SAVED DATASET")
print("-" * 70)

verification = pd.read_csv(
    OUTPUT_FILE
)

verification["hour_timestamp"] = pd.to_datetime(
    verification["hour_timestamp"]
)

print(
    f"Saved rows    : "
    f"{len(verification):,}"
)

print(
    f"Saved columns : "
    f"{len(verification.columns):,}"
)

print("\nSaved columns:")

for i, col in enumerate(
    verification.columns,
    start=1
):
    print(
        f"{i:2d}. {col}"
    )


# ================================================================
# 25. ORIGINAL DATASET INTEGRITY
# ================================================================

print("\n23. ORIGINAL DATASET INTEGRITY CHECK")
print("-" * 70)

raw_after = pd.read_csv(
    RAW_FILE
)

print(
    f"Original trip.csv rows: "
    f"{len(raw_after):,}"
)

print(
    f"Original trip.csv columns: "
    f"{len(raw_after.columns):,}"
)

if (
    len(raw_after) == original_rows
    and
    len(raw_after.columns) == original_columns
):

    print(
        "trip.csv remains unchanged."
    )

else:

    print(
        "WARNING: Raw dataset dimensions changed!"
    )


# ================================================================
# 26. PREPROCESSING SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

print(
    f"Raw trip records              : "
    f"{original_rows:,}"
)

print(
    f"Unique stations               : "
    f"{processed['start_station_id'].nunique():,}"
)

print(
    f"Hourly timestamps             : "
    f"{processed['hour_timestamp'].nunique():,}"
)

print(
    f"Final station-hour records    : "
    f"{len(processed):,}"
)

print(
    f"Positive-demand records       : "
    f"{positive_demand:,}"
)

print(
    f"Zero-demand records           : "
    f"{zero_demand:,}"
)

print(
    f"Maximum hourly station demand : "
    f"{processed['demand'].max()}"
)

print(
    f"Mean hourly station demand    : "
    f"{processed['demand'].mean():.4f}"
)

print(
    f"Final missing values          : "
    f"{processed.isnull().sum().sum():,}"
)

print(
    f"Final duplicate station-hours: "
    f"{station_hour_duplicates:,}"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\n")
print("=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNext step:")
print("Run Feature Engineering.")