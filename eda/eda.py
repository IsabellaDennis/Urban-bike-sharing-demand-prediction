# ================================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# EXPLORATORY DATA ANALYSIS
# Dataset: SF Bay Area Bike Share - trip.csv
# ================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ================================================================
# 1. PATH CONFIGURATION
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "trip.csv"
)

EDA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "eda"
)

PLOTS_DIR = os.path.join(
    EDA_DIR,
    "plots"
)

os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ================================================================
# 2. HEADER
# ================================================================

print("=" * 70)
print("URBAN BIKE-SHARING DEMAND PREDICTION")
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ================================================================
# 3. LOAD DATASET
# ================================================================

print("\n1. LOADING DATASET")
print("-" * 70)

if not os.path.exists(DATASET_PATH):
    print("ERROR: trip.csv not found.")
    print("Expected location:")
    print(DATASET_PATH)
    raise FileNotFoundError(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

# Store ORIGINAL dimensions before creating analysis columns
original_rows, original_columns = df.shape

print("Dataset loaded successfully.")
print(f"Rows    : {original_rows:,}")
print(f"Columns : {original_columns}")


# ================================================================
# 4. DATASET COLUMNS
# ================================================================

print("\n2. DATASET COLUMNS")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ================================================================
# 5. FIRST FIVE ROWS
# ================================================================

print("\n3. FIRST FIVE ROWS")
print("-" * 70)

print(df.head())


# ================================================================
# 6. DATA TYPES
# ================================================================

print("\n4. DATA TYPES")
print("-" * 70)

print(df.dtypes)


# ================================================================
# 7. MISSING VALUE ANALYSIS
# ================================================================

print("\n5. MISSING VALUE ANALYSIS")
print("-" * 70)

missing_count = df.isnull().sum()
missing_percentage = (missing_count / original_rows) * 100

missing_table = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": missing_count.values,
    "Missing Percentage": missing_percentage.values
})

print(missing_table.to_string(index=False))

total_missing = int(df.isnull().sum().sum())

print(f"\nTotal missing values: {total_missing:,}")


# Save missing-value report
missing_table.to_csv(
    os.path.join(EDA_DIR, "missing_values.csv"),
    index=False
)


# ================================================================
# 8. DUPLICATE RECORD ANALYSIS
# ================================================================

print("\n6. DUPLICATE RECORD ANALYSIS")
print("-" * 70)

duplicate_rows = int(df.duplicated().sum())

print(f"Duplicate rows: {duplicate_rows:,}")

if duplicate_rows == 0:
    print("No duplicate records found.")
else:
    print("Duplicate records detected.")


duplicate_summary = pd.DataFrame({
    "Metric": ["Total Rows", "Duplicate Rows"],
    "Count": [original_rows, duplicate_rows]
})

duplicate_summary.to_csv(
    os.path.join(EDA_DIR, "duplicate_analysis.csv"),
    index=False
)


# ================================================================
# 9. DATE CONVERSION
# ================================================================

print("\n7. DATE RANGE ANALYSIS")
print("-" * 70)

print("Converting date columns using mixed date format...")

# IMPORTANT:
# The original dataset contains dates such as:
# 8/29/2013 14:13
#
# format='mixed' correctly handles the date strings.

df["start_date_parsed"] = pd.to_datetime(
    df["start_date"],
    format="mixed",
    errors="coerce"
)

df["end_date_parsed"] = pd.to_datetime(
    df["end_date"],
    format="mixed",
    errors="coerce"
)

invalid_start_dates = int(df["start_date_parsed"].isna().sum())
invalid_end_dates = int(df["end_date_parsed"].isna().sum())

print(f"Invalid start dates: {invalid_start_dates:,}")
print(f"Invalid end dates  : {invalid_end_dates:,}")

if invalid_start_dates == 0:
    print("All start dates parsed successfully.")

if invalid_end_dates == 0:
    print("All end dates parsed successfully.")


print("\nStart date:")
print(df["start_date_parsed"].min())

print("\nEnd date:")
print(df["start_date_parsed"].max())


# ================================================================
# 10. NUMERICAL STATISTICS
# ================================================================

print("\n8. NUMERICAL STATISTICS")
print("-" * 70)

print(df.describe(include="all").transpose())


# Save numerical statistics
numeric_statistics = df.describe(
    include="all"
).transpose()

numeric_statistics.to_csv(
    os.path.join(EDA_DIR, "statistical_summary.csv")
)


# ================================================================
# 11. UNIQUE VALUE ANALYSIS
# ================================================================

print("\n9. UNIQUE VALUE ANALYSIS")
print("-" * 70)

unique_table = []

for column in df.columns[:original_columns]:

    unique_count = df[column].nunique(dropna=True)

    unique_table.append({
        "Column": column,
        "Unique Values": unique_count
    })

    print(f"{column:22} : {unique_count:,} unique values")


unique_table = pd.DataFrame(unique_table)

unique_table.to_csv(
    os.path.join(EDA_DIR, "unique_values.csv"),
    index=False
)


# ================================================================
# 12. STATION ANALYSIS
# ================================================================

print("\n10. STATION ANALYSIS")
print("-" * 70)

unique_start_stations = df["start_station_id"].nunique()
unique_start_names = df["start_station_name"].nunique()
unique_end_stations = df["end_station_id"].nunique()
unique_end_names = df["end_station_name"].nunique()

print(f"Unique start stations      : {unique_start_stations}")
print(f"Unique start station names : {unique_start_names}")
print(f"Unique end stations        : {unique_end_stations}")
print(f"Unique end station names   : {unique_end_names}")


# ================================================================
# 13. SUBSCRIPTION TYPE ANALYSIS
# ================================================================

print("\n11. SUBSCRIPTION TYPE ANALYSIS")
print("-" * 70)

subscription_counts = (
    df["subscription_type"]
    .value_counts()
)

print(subscription_counts)


subscription_counts.to_csv(
    os.path.join(EDA_DIR, "subscription_type.csv")
)


# ================================================================
# 14. TRIP DURATION ANALYSIS
# ================================================================

print("\n12. TRIP DURATION ANALYSIS")
print("-" * 70)

duration_statistics = df["duration"].describe()

print(duration_statistics)


# Duration in minutes
df["duration_minutes"] = df["duration"] / 60

print("\nDuration in minutes:")
print(df["duration_minutes"].describe())


# Identify extreme duration values
duration_q1 = df["duration"].quantile(0.25)
duration_q3 = df["duration"].quantile(0.75)

duration_iqr = duration_q3 - duration_q1

duration_upper_limit = duration_q3 + (1.5 * duration_iqr)

duration_outliers = int(
    (df["duration"] > duration_upper_limit).sum()
)

print(f"\nIQR upper limit: {duration_upper_limit:,.2f} seconds")
print(f"Potential duration outliers: {duration_outliers:,}")


duration_summary = pd.DataFrame({
    "Metric": [
        "Mean Duration (seconds)",
        "Median Duration (seconds)",
        "Minimum Duration (seconds)",
        "Maximum Duration (seconds)",
        "Potential Outliers"
    ],
    "Value": [
        df["duration"].mean(),
        df["duration"].median(),
        df["duration"].min(),
        df["duration"].max(),
        duration_outliers
    ]
})

duration_summary.to_csv(
    os.path.join(EDA_DIR, "duration_analysis.csv"),
    index=False
)


# ================================================================
# 15. CREATE TEMPORARY TIME FEATURES
# ================================================================

print("\n13. TEMPORAL DEMAND ANALYSIS")
print("-" * 70)

# These are ANALYSIS columns only.
# They do NOT change the original trip.csv.

df["hour"] = df["start_date_parsed"].dt.hour
df["weekday"] = df["start_date_parsed"].dt.weekday
df["weekday_name"] = df["start_date_parsed"].dt.day_name()
df["month"] = df["start_date_parsed"].dt.month
df["year"] = df["start_date_parsed"].dt.year
df["date"] = df["start_date_parsed"].dt.date


# ================================================================
# 16. HOURLY DEMAND
# ================================================================

hourly_demand = (
    df.groupby("hour")
      .size()
      .reset_index(name="demand")
)

print("\nHourly demand:")
print(hourly_demand.to_string(index=False))


hourly_demand.to_csv(
    os.path.join(EDA_DIR, "hourly_demand.csv"),
    index=False
)


# Plot hourly demand
plt.figure(figsize=(10, 5))

plt.plot(
    hourly_demand["hour"],
    hourly_demand["demand"],
    marker="o"
)

plt.title("Hourly Bike-Sharing Demand")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.xticks(range(24))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "hourly_demand.png"),
    dpi=300
)

plt.close()


# ================================================================
# 17. DAILY DEMAND
# ================================================================

daily_demand = (
    df.groupby("date")
      .size()
      .reset_index(name="demand")
)

print("\nDaily demand statistics:")
print(daily_demand["demand"].describe())


daily_demand.to_csv(
    os.path.join(EDA_DIR, "daily_demand.csv"),
    index=False
)


# Plot daily demand
plt.figure(figsize=(12, 5))

plt.plot(
    pd.to_datetime(daily_demand["date"]),
    daily_demand["demand"]
)

plt.title("Daily Bike-Sharing Demand")
plt.xlabel("Date")
plt.ylabel("Number of Trips")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "daily_demand.png"),
    dpi=300
)

plt.close()


# ================================================================
# 18. MONTHLY DEMAND
# ================================================================

monthly_demand = (
    df.groupby("month")
      .size()
      .reset_index(name="demand")
)

print("\nMonthly demand:")
print(monthly_demand.to_string(index=False))


monthly_demand.to_csv(
    os.path.join(EDA_DIR, "monthly_demand.csv"),
    index=False
)


# Plot monthly demand
plt.figure(figsize=(10, 5))

plt.bar(
    monthly_demand["month"],
    monthly_demand["demand"]
)

plt.title("Monthly Bike-Sharing Demand")
plt.xlabel("Month")
plt.ylabel("Number of Trips")
plt.xticks(range(1, 13))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "monthly_demand.png"),
    dpi=300
)

plt.close()


# ================================================================
# 19. WEEKDAY DEMAND
# ================================================================

weekday_demand = (
    df.groupby(["weekday", "weekday_name"])
      .size()
      .reset_index(name="demand")
      .sort_values("weekday")
)

print("\nWeekday demand:")
print(weekday_demand.to_string(index=False))


weekday_demand.to_csv(
    os.path.join(EDA_DIR, "weekday_demand.csv"),
    index=False
)


# Plot weekday demand
plt.figure(figsize=(10, 5))

plt.bar(
    weekday_demand["weekday_name"],
    weekday_demand["demand"]
)

plt.title("Weekday Bike-Sharing Demand")
plt.xlabel("Day of Week")
plt.ylabel("Number of Trips")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "weekday_demand.png"),
    dpi=300
)

plt.close()


# ================================================================
# 20. STATION-WISE DEMAND
# ================================================================

station_demand = (
    df.groupby(
        ["start_station_id", "start_station_name"]
    )
    .size()
    .reset_index(name="demand")
    .sort_values(
        "demand",
        ascending=False
    )
)

print("\nTop 20 stations by trip demand:")

print(
    station_demand
    .head(20)
    .to_string(index=False)
)


station_demand.to_csv(
    os.path.join(EDA_DIR, "station_demand.csv"),
    index=False
)


# Plot top 20 stations
top20 = station_demand.head(20).sort_values("demand")

plt.figure(figsize=(10, 8))

plt.barh(
    top20["start_station_name"],
    top20["demand"]
)

plt.title("Top 20 Stations by Trip Demand")
plt.xlabel("Number of Trips")
plt.ylabel("Station")
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "top20_stations.png"),
    dpi=300
)

plt.close()


# ================================================================
# 21. START STATION VS END STATION ANALYSIS
# ================================================================

print("\n14. START / END STATION ANALYSIS")
print("-" * 70)

start_station_counts = (
    df["start_station_id"]
    .value_counts()
    .sort_values(ascending=False)
)

end_station_counts = (
    df["end_station_id"]
    .value_counts()
    .sort_values(ascending=False)
)

station_flow = pd.DataFrame({
    "start_trip_count": start_station_counts,
    "end_trip_count": end_station_counts
}).fillna(0)

station_flow["net_flow"] = (
    station_flow["start_trip_count"]
    - station_flow["end_trip_count"]
)

station_flow.index.name = "station_id"

print("\nTop stations by net outgoing demand:")

print(
    station_flow
    .sort_values(
        "net_flow",
        ascending=False
    )
    .head(10)
)


station_flow.to_csv(
    os.path.join(EDA_DIR, "station_flow_analysis.csv")
)


# ================================================================
# 22. SUBSCRIPTION TYPE VS DEMAND
# ================================================================

subscription_demand = (
    df.groupby("subscription_type")
      .size()
      .reset_index(name="demand")
)

subscription_demand.to_csv(
    os.path.join(EDA_DIR, "subscription_demand.csv"),
    index=False
)


# ================================================================
# 23. WEEKDAY-HOUR DEMAND
# ================================================================

weekday_hour_demand = (
    df.groupby(
        ["weekday", "weekday_name", "hour"]
    )
    .size()
    .reset_index(name="demand")
)

weekday_hour_demand.to_csv(
    os.path.join(EDA_DIR, "weekday_hour_demand.csv"),
    index=False
)


# ================================================================
# 24. PEAK HOUR ANALYSIS
# ================================================================

print("\n15. PEAK HOUR ANALYSIS")
print("-" * 70)

peak_hour = hourly_demand.loc[
    hourly_demand["demand"].idxmax()
]

low_hour = hourly_demand.loc[
    hourly_demand["demand"].idxmin()
]

print(
    f"Highest demand hour : "
    f"{int(peak_hour['hour']):02d}:00 "
    f"({int(peak_hour['demand']):,} trips)"
)

print(
    f"Lowest demand hour  : "
    f"{int(low_hour['hour']):02d}:00 "
    f"({int(low_hour['demand']):,} trips)"
)


# ================================================================
# 25. WEEKDAY VS WEEKEND
# ================================================================

df["day_type"] = np.where(
    df["weekday"] < 5,
    "Weekday",
    "Weekend"
)

day_type_demand = (
    df.groupby("day_type")
      .size()
      .reset_index(name="demand")
)

print("\nWeekday vs Weekend demand:")
print(day_type_demand.to_string(index=False))

day_type_demand.to_csv(
    os.path.join(EDA_DIR, "weekday_vs_weekend.csv"),
    index=False
)


# ================================================================
# 26. CORRELATION ANALYSIS
# ================================================================

print("\n16. NUMERICAL FEATURE CORRELATION")
print("-" * 70)

correlation_columns = [
    "duration",
    "start_station_id",
    "end_station_id",
    "bike_id",
    "hour",
    "weekday",
    "month",
    "year"
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
)

print(correlation_matrix.round(4))


correlation_matrix.to_csv(
    os.path.join(EDA_DIR, "correlation_matrix.csv")
)


# Correlation heatmap using matplotlib only
plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_columns)),
    correlation_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_columns)),
    correlation_columns
)

plt.title("Numerical Feature Correlation")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "correlation_matrix.png"),
    dpi=300
)

plt.close()


# ================================================================
# 27. DATA QUALITY SUMMARY
# ================================================================

print("\n17. DATA QUALITY SUMMARY")
print("-" * 70)

quality_summary = pd.DataFrame({
    "Metric": [
        "Total trip records",
        "Total raw columns",
        "Missing values",
        "Duplicate rows",
        "Invalid start dates",
        "Invalid end dates",
        "Unique start stations",
        "Unique end stations",
        "Maximum trip duration (seconds)",
        "Minimum trip duration (seconds)"
    ],
    "Value": [
        original_rows,
        original_columns,
        total_missing,
        duplicate_rows,
        invalid_start_dates,
        invalid_end_dates,
        unique_start_stations,
        unique_end_stations,
        df["duration"].max(),
        df["duration"].min()
    ]
})

print(quality_summary.to_string(index=False))

quality_summary.to_csv(
    os.path.join(EDA_DIR, "eda_quality_summary.csv"),
    index=False
)


# ================================================================
# 28. SPECIAL EVENT DEMAND ANALYSIS
# ================================================================

print("\n18. SPECIAL EVENT DEMAND ANALYSIS")
print("-" * 70)

if "special_event" in df.columns:
    event_demand = df.groupby("special_event").size().reset_index(name="demand")
    event_demand["percentage"] = (event_demand["demand"] / len(df)) * 100
    event_demand = event_demand.sort_values("demand", ascending=False).reset_index(drop=True)
    print(event_demand.to_string(index=False))
    event_demand.to_csv(os.path.join(EDA_DIR, "special_event_demand.csv"), index=False)

    df["is_event"] = (df["special_event"] != "No Event").astype(int)
    event_vs_no_event = df.groupby("is_event").size().reset_index(name="demand")
    event_vs_no_event["type"] = event_vs_no_event["is_event"].map({0: "No Event", 1: "Special Event"})
    event_vs_no_event["percentage"] = (event_vs_no_event["demand"] / len(df)) * 100
    event_vs_no_event.to_csv(os.path.join(EDA_DIR, "event_vs_no_event_demand.csv"), index=False)


# ================================================================
# 29. EDA SUMMARY
# ================================================================

print("\n19. EDA SUMMARY")
print("-" * 70)

print(
    f"Total trip records       : {original_rows:,}"
)

print(
    f"Total raw columns        : {original_columns}"
)

print(
    f"Unique start stations    : {unique_start_stations}"
)

print(
    f"Unique end stations      : {unique_end_stations}"
)

print(
    f"Date range               : "
    f"{df['start_date_parsed'].min()} "
    f"to "
    f"{df['start_date_parsed'].max()}"
)

print(
    f"Total duplicate rows     : {duplicate_rows:,}"
)

print(
    f"Total missing values     : {total_missing:,}"
)

print(
    f"Invalid start dates      : {invalid_start_dates:,}"
)

print(
    f"Invalid end dates        : {invalid_end_dates:,}"
)

print(
    f"Maximum demand hour      : "
    f"{int(peak_hour['hour']):02d}:00"
)

print(
    f"Maximum trip duration    : "
    f"{df['duration'].max():,.0f} seconds"
)


# ================================================================
# 29. SAVE EDA RESULTS
# ================================================================

print("\n19. SAVING EDA RESULTS")
print("-" * 70)

print("EDA tables saved successfully.")
print("EDA plots saved successfully.")

print("\nEDA output directory:")
print(EDA_DIR)


# ================================================================
# 30. IMPORTANT RAW DATA CHECK
# ================================================================

print("\n20. ORIGINAL DATASET INTEGRITY CHECK")
print("-" * 70)

print(
    f"Original dataset dimensions: "
    f"{original_rows:,} rows × {original_columns} columns"
)

print(
    "Temporary EDA columns were NOT saved back to trip.csv."
)

print(
    "Raw dataset remains unchanged."
)


# ================================================================
# 31. COMPLETION
# ================================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNext step:")
print("Run preprocessing.py")

print("=" * 70)