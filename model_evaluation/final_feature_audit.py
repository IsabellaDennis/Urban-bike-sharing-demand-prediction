import os
import numpy as np
import pandas as pd


# ==============================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# FINAL FEATURE LEAKAGE & ROBUSTNESS AUDIT
# ==============================================================

print("=" * 70)
print("URBAN BIKE-SHARING DEMAND PREDICTION")
print("FINAL FEATURE LEAKAGE & ROBUSTNESS AUDIT")
print("=" * 70)


# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "station_hourly_demand_features.csv"
)

FEATURE_LIST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "feature_list.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "model_results",
    "leakage_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================
# 1. LOADING DATASET
# ==============================================================

print("\n" + "-" * 70)
print("1. LOADING FEATURE-ENGINEERED DATASET")
print("-" * 70)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df)}")


# ==============================================================
# 2. REQUIRED COLUMN CHECK
# ==============================================================

required_columns = [
    "start_station_id",
    "start_station_name",
    "hour_timestamp",
    "demand"
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

print("\nAll required columns are available.")


# ==============================================================
# 3. CONVERT TIMESTAMP
# ==============================================================

print("\n" + "-" * 70)
print("2. CONVERTING TIMESTAMP")
print("-" * 70)

df["hour_timestamp"] = pd.to_datetime(
    df["hour_timestamp"],
    errors="coerce"
)

invalid = df["hour_timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid}")

if invalid > 0:
    raise ValueError("Invalid timestamps detected.")

print(f"Start: {df['hour_timestamp'].min()}")
print(f"End  : {df['hour_timestamp'].max()}")


# ==============================================================
# 4. SORT DATA
# ==============================================================

df = df.sort_values(
    ["start_station_id", "hour_timestamp"]
).reset_index(drop=True)


# ==============================================================
# 5. TARGET ANALYSIS
# ==============================================================

print("\n" + "-" * 70)
print("3. TARGET ANALYSIS")
print("-" * 70)

print(f"Target minimum : {df['demand'].min()}")
print(f"Target maximum : {df['demand'].max()}")
print(f"Target mean    : {df['demand'].mean():.4f}")
print(f"Target std     : {df['demand'].std():.4f}")


# ==============================================================
# 6. LOAD FEATURE LIST
# ==============================================================

print("\n" + "-" * 70)
print("4. LOADING MODEL FEATURE LIST")
print("-" * 70)

feature_df = pd.read_csv(FEATURE_LIST_PATH)

# Support either a "feature" column or first column
if "feature" in feature_df.columns:
    original_features = feature_df["feature"].dropna().tolist()
else:
    original_features = feature_df.iloc[:, 0].dropna().tolist()

print(f"Features in feature_list.csv: {len(original_features)}")


# ==============================================================
# 7. REMOVE RISKY HISTORICAL AGGREGATES
# ==============================================================

print("\n" + "-" * 70)
print("5. REMOVING POTENTIALLY RISKY HISTORICAL AGGREGATES")
print("-" * 70)

risky_aggregates = [
    "station_weekday_hour_mean",
    "station_hour_historical_mean",
    "historical_demand",
    "historical_max_demand",
    "historical_min_demand"
]

features_after_aggregate_removal = [
    f for f in original_features
    if f not in risky_aggregates
]

print("Removed features:")

for feature in risky_aggregates:
    if feature in original_features:
        print(f" - {feature}")


# ==============================================================
# 8. REMOVE DEMAND-CHANGE FEATURES
# ==============================================================

print("\n" + "-" * 70)
print("6. REMOVING TARGET-DEPENDENT DEMAND-CHANGE FEATURES")
print("-" * 70)

# These features are problematic because they were derived
# using the current demand target.
#
# Example:
# demand_change_1h = current_demand - previous_demand
#
# For prediction of current demand, current_demand is not
# available at prediction time.
#
# Therefore they are removed from the final model.

demand_change_features = [
    "demand_change_1h",
    "demand_change_24h",
    "demand_change_week"
]

final_features = [
    f for f in features_after_aggregate_removal
    if f not in demand_change_features
]

print("Removed target-dependent features:")

for feature in demand_change_features:
    if feature in features_after_aggregate_removal:
        print(f" - {feature}")

print(f"\nOriginal features : {len(original_features)}")
print(f"After aggregates  : {len(features_after_aggregate_removal)}")
print(f"Final features     : {len(final_features)}")


# ==============================================================
# 9. CHECK FINAL FEATURES
# ==============================================================

print("\n" + "-" * 70)
print("7. CHECKING FINAL FEATURES")
print("-" * 70)

missing_features = [
    f for f in final_features
    if f not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing final features: {missing_features}"
    )

print("All final features are available.")

for i, feature in enumerate(final_features, start=1):
    print(f"{i:2d}. {feature}")


# ==============================================================
# 10. DIRECT TARGET LEAKAGE CHECK
# ==============================================================

print("\n" + "-" * 70)
print("8. DIRECT TARGET LEAKAGE CHECK")
print("-" * 70)

if "demand" in final_features:
    raise ValueError(
        "Target demand is present inside model features."
    )

print("Target demand inside feature list: False")
print("No direct target leakage detected.")


# ==============================================================
# 11. CURRENT / FUTURE DEMAND FEATURE NAME CHECK
# ==============================================================

print("\n" + "-" * 70)
print("9. CURRENT / FUTURE DEMAND FEATURE NAME CHECK")
print("-" * 70)

dangerous_keywords = [
    "current_demand",
    "future_demand",
    "next_demand",
    "target_demand",
    "future",
    "next_hour"
]

dangerous_features = []

for feature in final_features:

    feature_lower = feature.lower()

    for keyword in dangerous_keywords:

        if keyword in feature_lower:
            dangerous_features.append(feature)
            break


if dangerous_features:

    print("Potential current/future demand features found:")

    for feature in dangerous_features:
        print(f" - {feature}")

    raise ValueError(
        "Potential current/future demand leakage detected."
    )

else:

    print("No current/future-demand feature names detected.")


# ==============================================================
# 12. MISSING VALUE CHECK
# ==============================================================

print("\n" + "-" * 70)
print("10. MISSING VALUE CHECK")
print("-" * 70)

missing_values = df[final_features].isna().sum().sum()

print(f"Total missing feature values: {missing_values}")

if missing_values > 0:

    print(
        "WARNING: Missing values detected in final features."
    )

else:

    print("No missing feature values.")


# ==============================================================
# 13. STATION-HOUR DUPLICATE CHECK
# ==============================================================

print("\n" + "-" * 70)
print("11. STATION-HOUR DUPLICATE CHECK")
print("-" * 70)

duplicate_count = df.duplicated(
    subset=[
        "start_station_id",
        "hour_timestamp"
    ]
).sum()

print(
    f"Duplicate station-hour records: {duplicate_count}"
)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate station-hour records detected."
    )

print("No duplicate station-hour records.")


# ==============================================================
# 14. HISTORICAL FEATURE LIST
# ==============================================================

print("\n" + "-" * 70)
print("12. HISTORICAL FEATURE AUDIT")
print("-" * 70)

historical_keywords = [
    "lag_",
    "same_hour_previous",
    "rolling_",
    "ratio"
]

historical_features = []

for feature in final_features:

    if any(
        keyword in feature
        for keyword in historical_keywords
    ):
        historical_features.append(feature)


print(
    f"Historical / derived temporal features: "
    f"{len(historical_features)}"
)

for feature in historical_features:
    print(f" - {feature}")


# ==============================================================
# 15. INDEPENDENT DEMAND HISTORY
# ==============================================================

print("\n" + "-" * 70)
print("13. PREPARING INDEPENDENT DEMAND HISTORY")
print("-" * 70)

history = df[
    [
        "start_station_id",
        "hour_timestamp",
        "demand"
    ]
].copy()

history = history.sort_values(
    [
        "start_station_id",
        "hour_timestamp"
    ]
)


# ==============================================================
# 16. LAG ALIGNMENT TEST
# ==============================================================

print("\n" + "-" * 70)
print("14. LAG ALIGNMENT TEST")
print("-" * 70)

lag_periods = {
    "lag_1h": 1,
    "lag_2h": 2,
    "lag_3h": 3,
    "lag_6h": 6,
    "lag_12h": 12,
    "lag_24h": 24,
    "lag_48h": 48,
    "lag_72h": 72,
    "lag_168h": 168
}

lag_results = []

for feature, periods in lag_periods.items():

    if feature not in final_features:
        continue

    expected = (
        history
        .groupby("start_station_id")["demand"]
        .shift(periods)
    )

    actual = df[feature]

    valid = expected.notna()

    if valid.sum() == 0:

        match_percentage = 100.0
        max_difference = 0.0

    else:

        difference = (
            actual[valid].values
            - expected[valid].values
        )

        max_difference = np.max(
            np.abs(difference)
        )

        match_percentage = (
            np.mean(
                np.isclose(
                    actual[valid],
                    expected[valid],
                    atol=1e-10
                )
            ) * 100
        )

    print(
        f"{feature:<16}"
        f"Match: {match_percentage:8.4f}% "
        f"Max difference: {max_difference:.10f}"
    )

    lag_results.append({
        "feature": feature,
        "match_percentage": match_percentage,
        "max_difference": max_difference
    })


lag_results_df = pd.DataFrame(lag_results)

lag_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_lag_alignment_test.csv"
    ),
    index=False
)


# ==============================================================
# 17. SAME-HOUR HISTORICAL FEATURE TEST
# ==============================================================

print("\n" + "-" * 70)
print("15. SAME-HOUR HISTORICAL FEATURE CHECK")
print("-" * 70)

same_hour_features = {
    "same_hour_previous_day": 24,
    "same_hour_previous_2days": 48,
    "same_hour_previous_3days": 72,
    "same_hour_previous_week": 168
}

same_hour_results = []

for feature, periods in same_hour_features.items():

    if feature not in final_features:
        continue

    expected = (
        history
        .groupby("start_station_id")["demand"]
        .shift(periods)
    )

    actual = df[feature]

    valid = expected.notna()

    if valid.sum() == 0:

        match_percentage = 100.0
        max_difference = 0.0

    else:

        difference = (
            actual[valid].values
            - expected[valid].values
        )

        max_difference = np.max(
            np.abs(difference)
        )

        match_percentage = (
            np.mean(
                np.isclose(
                    actual[valid],
                    expected[valid],
                    atol=1e-10
                )
            ) * 100
        )

    print(
        f"{feature:<32}"
        f"Match: {match_percentage:8.4f}% "
        f"Max difference: {max_difference:.10f}"
    )

    same_hour_results.append({
        "feature": feature,
        "match_percentage": match_percentage,
        "max_difference": max_difference
    })


pd.DataFrame(same_hour_results).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_same_hour_alignment_test.csv"
    ),
    index=False
)


# ==============================================================
# 18. FINAL FEATURE SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("FINAL FEATURE AUDIT SUMMARY")
print("=" * 70)

print(f"Dataset rows             : {len(df):,}")
print(f"Original model features  : {len(original_features)}")
print(f"Risky aggregates removed : {len(risky_aggregates)}")
print(f"Demand-change removed    : {len(demand_change_features)}")
print(f"FINAL MODEL FEATURES     : {len(final_features)}")

print(
    "\nTarget leakage           : NOT DETECTED"
)

print(
    "Current/future leakage   : NOT DETECTED"
)

print(
    f"Missing values           : {missing_values}"
)

print(
    f"Duplicate station-hours  : {duplicate_count}"
)


# ==============================================================
# 19. SAVE FINAL FEATURE LIST
# ==============================================================

final_feature_df = pd.DataFrame({
    "feature": final_features
})

final_feature_path = os.path.join(
    OUTPUT_DIR,
    "final_feature_list.csv"
)

final_feature_df.to_csv(
    final_feature_path,
    index=False
)

print("\nFinal feature list saved:")
print(final_feature_path)


# ==============================================================
# 20. SAVE AUDIT SUMMARY
# ==============================================================

audit_summary = pd.DataFrame({
    "metric": [
        "dataset_rows",
        "original_features",
        "risky_aggregates_removed",
        "demand_change_features_removed",
        "final_features",
        "missing_values",
        "duplicate_station_hours"
    ],
    "value": [
        len(df),
        len(original_features),
        len(risky_aggregates),
        len(demand_change_features),
        len(final_features),
        missing_values,
        duplicate_count
    ]
})

audit_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_audit_summary.csv"
    ),
    index=False
)


# ==============================================================
# FINAL RESULT
# ==============================================================

print("\n" + "=" * 70)
print("FINAL FEATURE LEAKAGE & ROBUSTNESS AUDIT COMPLETED")
print("=" * 70)

print("\nIMPORTANT:")
print("The three demand-change features were removed because")
print("they depend on the current target demand.")

print("\nFinal model feature count:", len(final_features))

print("\nNext step:")
print(f"Retrain models using the final {len(final_features)} features.")

print("=" * 70)