# ================================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# FEATURE LEAKAGE & ROBUSTNESS AUDIT
# ================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "station_hourly_demand_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "model_results",
    "leakage_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# DISPLAY
# ================================================================

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# 1. LOAD DATA
# ================================================================

section("URBAN BIKE-SHARING DEMAND PREDICTION")
print("FEATURE LEAKAGE & ROBUSTNESS AUDIT")

section("1. LOADING FEATURE-ENGINEERED DATASET")

df = pd.read_csv(FEATURE_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ================================================================
# 2. TIMESTAMP
# ================================================================

section("2. CONVERTING TIMESTAMP")

df["hour_timestamp"] = pd.to_datetime(
    df["hour_timestamp"],
    errors="coerce"
)

invalid = df["hour_timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid}")

if invalid > 0:
    raise ValueError("Invalid timestamps detected.")

df = df.sort_values(
    ["start_station_id", "hour_timestamp"]
).reset_index(drop=True)

print(f"Start: {df['hour_timestamp'].min()}")
print(f"End  : {df['hour_timestamp'].max()}")


# ================================================================
# 3. TARGET
# ================================================================

TARGET = "demand"

print("\nTarget statistics:")
print(f"Minimum : {df[TARGET].min()}")
print(f"Maximum : {df[TARGET].max()}")
print(f"Mean    : {df[TARGET].mean():.4f}")
print(f"Std     : {df[TARGET].std():.4f}")


# ================================================================
# 4. IDENTIFY FEATURES
# ================================================================

EXCLUDE_COLUMNS = [
    "demand",
    "hour_timestamp",
    "start_station_name"
]

FEATURES = [
    col for col in df.columns
    if col not in EXCLUDE_COLUMNS
]

print("\nTotal model features:", len(FEATURES))


# ================================================================
# 5. HISTORICAL FEATURE CLASSIFICATION
# ================================================================

section("3. HISTORICAL FEATURE AUDIT")

historical_keywords = [
    "lag",
    "rolling",
    "historical",
    "previous",
    "change",
    "ratio",
    "mean",
    "max",
    "min",
    "std",
    "median"
]

historical_features = []

for feature in FEATURES:
    feature_lower = feature.lower()

    if any(keyword in feature_lower for keyword in historical_keywords):
        historical_features.append(feature)

print(
    f"Historical / derived temporal features detected: "
    f"{len(historical_features)}"
)

for feature in historical_features:
    print(" -", feature)


# ================================================================
# 6. EXPLICIT RISKY FEATURES
# ================================================================

section("4. POTENTIALLY RISKY HISTORICAL AGGREGATES")

risky_features = [
    "station_weekday_hour_mean",
    "station_hour_historical_mean",
    "historical_demand",
    "historical_max_demand",
    "historical_min_demand"
]

risky_features = [
    feature for feature in risky_features
    if feature in df.columns
]

print(f"Potentially risky features found: {len(risky_features)}")

for feature in risky_features:
    print(" -", feature)


# ================================================================
# 7. FEATURE AVAILABILITY TEST
# ================================================================

section("5. HISTORICAL FEATURE AVAILABILITY TEST")

audit_results = []

for feature in historical_features:

    if feature not in df.columns:
        continue

    non_null = df[feature].notna().sum()

    audit_results.append({
        "feature": feature,
        "non_null_values": non_null,
        "missing_values": df[feature].isna().sum(),
        "minimum": df[feature].min(),
        "maximum": df[feature].max(),
        "mean": df[feature].mean()
    })

audit_df = pd.DataFrame(audit_results)

audit_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "historical_feature_audit.csv"
    ),
    index=False
)

print("Historical feature audit saved.")


# ================================================================
# 8. CHECK LAG ALIGNMENT
# ================================================================

section("6. LAG ALIGNMENT TEST")

lag_tests = []

grouped = df.groupby("start_station_id", sort=False)

for lag_hours in [1, 2, 3, 6, 12, 24, 48, 72, 168]:

    column = f"lag_{lag_hours}h"

    if column not in df.columns:
        continue

    expected = grouped[TARGET].shift(lag_hours)

    valid = expected.notna()

    matches = np.isclose(
        df.loc[valid, column].values,
        expected.loc[valid].values,
        equal_nan=False
    )

    match_percentage = (
        matches.mean() * 100
        if len(matches) > 0
        else 0
    )

    lag_tests.append({
        "lag_feature": column,
        "expected_lag_hours": lag_hours,
        "matching_percentage": match_percentage
    })

    print(
        f"{column:<15} "
        f"Match: {match_percentage:.4f}%"
    )

lag_df = pd.DataFrame(lag_tests)

lag_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lag_alignment_test.csv"
    ),
    index=False
)


# ================================================================
# 9. CHECK ROLLING FEATURE ALIGNMENT
# ================================================================

section("7. ROLLING FEATURE ALIGNMENT TEST")

rolling_tests = []

for window in [3, 6, 12, 24, 168]:

    column = f"rolling_mean_{window}h"

    if column not in df.columns:
        continue

    expected = (
        df.groupby("start_station_id")[TARGET]
        .transform(
            lambda x: x.shift(1).rolling(
                window=window,
                min_periods=1
            ).mean()
        )
    )

    valid = expected.notna()

    difference = np.abs(
        df.loc[valid, column].values -
        expected.loc[valid].values
    )

    max_difference = difference.max()

    rolling_tests.append({
        "rolling_feature": column,
        "window": window,
        "maximum_difference": max_difference
    })

    print(
        f"{column:<22} "
        f"Maximum difference: {max_difference:.10f}"
    )

rolling_df = pd.DataFrame(rolling_tests)

rolling_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rolling_alignment_test.csv"
    ),
    index=False
)


# ================================================================
# 10. CHRONOLOGICAL SPLIT
# ================================================================

section("8. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")

timestamps = df["hour_timestamp"]

train_end = timestamps.quantile(0.70)
validation_end = timestamps.quantile(0.85)

train_mask = timestamps <= train_end

validation_mask = (
    (timestamps > train_end) &
    (timestamps <= validation_end)
)

test_mask = timestamps > validation_end

print("\nTRAIN")
print(f"Rows : {train_mask.sum():,}")
print(
    f"Time : {timestamps[train_mask].min()} "
    f"to {timestamps[train_mask].max()}"
)

print("\nVALIDATION")
print(f"Rows : {validation_mask.sum():,}")
print(
    f"Time : {timestamps[validation_mask].min()} "
    f"to {timestamps[validation_mask].max()}"
)

print("\nTEST")
print(f"Rows : {test_mask.sum():,}")
print(
    f"Time : {timestamps[test_mask].min()} "
    f"to {timestamps[test_mask].max()}"
)


# ================================================================
# 11. LEAKAGE CHECK FOR AGGREGATE FEATURES
# ================================================================

section("9. TEMPORAL AGGREGATE LEAKAGE CHECK")

aggregate_results = []

for feature in risky_features:

    if feature not in df.columns:
        continue

    train_values = df.loc[train_mask, feature]
    validation_values = df.loc[validation_mask, feature]
    test_values = df.loc[test_mask, feature]

    aggregate_results.append({
        "feature": feature,
        "train_mean": train_values.mean(),
        "validation_mean": validation_values.mean(),
        "test_mean": test_values.mean(),
        "train_max": train_values.max(),
        "validation_max": validation_values.max(),
        "test_max": test_values.max()
    })

aggregate_df = pd.DataFrame(aggregate_results)

print(aggregate_df.to_string(index=False))

aggregate_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aggregate_feature_audit.csv"
    ),
    index=False
)


# ================================================================
# 12. MODEL FUNCTION
# ================================================================

def train_and_evaluate(
    name,
    feature_list
):

    print("\n" + "-" * 70)
    print(f"MODEL: {name}")
    print("-" * 70)

    X_train = df.loc[train_mask, feature_list]
    y_train = df.loc[train_mask, TARGET]

    X_validation = df.loc[
        validation_mask,
        feature_list
    ]

    y_validation = df.loc[
        validation_mask,
        TARGET
    ]

    X_test = df.loc[test_mask, feature_list]
    y_test = df.loc[test_mask, TARGET]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    validation_pred = model.predict(X_validation)
    test_pred = model.predict(X_test)

    validation_mae = mean_absolute_error(
        y_validation,
        validation_pred
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_pred
        )
    )

    validation_r2 = r2_score(
        y_validation,
        validation_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )

    test_r2 = r2_score(
        y_test,
        test_pred
    )

    print(
        f"Validation MAE  : {validation_mae:.6f}"
    )

    print(
        f"Validation RMSE : {validation_rmse:.6f}"
    )

    print(
        f"Validation R²   : {validation_r2:.6f}"
    )

    print()

    print(
        f"Test MAE        : {test_mae:.6f}"
    )

    print(
        f"Test RMSE       : {test_rmse:.6f}"
    )

    print(
        f"Test R²         : {test_r2:.6f}"
    )

    return {
        "Model": name,
        "Features": len(feature_list),
        "Validation MAE": validation_mae,
        "Validation RMSE": validation_rmse,
        "Validation R2": validation_r2,
        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Test R2": test_r2
    }


# ================================================================
# 13. ROBUSTNESS EXPERIMENTS
# ================================================================

section("10. ROBUSTNESS EXPERIMENTS")

# ------------------------------------------------
# MODEL 1: ALL FEATURES
# ------------------------------------------------

results = []

results.append(
    train_and_evaluate(
        "XGBoost - All Features",
        FEATURES
    )
)


# ------------------------------------------------
# MODEL 2: REMOVE RISKY AGGREGATES
# ------------------------------------------------

features_without_risky = [
    feature
    for feature in FEATURES
    if feature not in risky_features
]

results.append(
    train_and_evaluate(
        "XGBoost - Without Historical Aggregates",
        features_without_risky
    )
)


# ------------------------------------------------
# MODEL 3: CALENDAR + LAGS
# ------------------------------------------------

calendar_features = [
    "start_station_id",
    "hour",
    "day",
    "weekday",
    "month",
    "year",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "is_weekday",
    "morning_peak",
    "evening_peak",
    "peak_hour",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos"
]

lag_features = [
    feature
    for feature in FEATURES
    if feature.startswith("lag_")
    or feature.startswith("same_hour_previous")
]

calendar_lag_features = [
    feature
    for feature in calendar_features + lag_features
    if feature in FEATURES
]

results.append(
    train_and_evaluate(
        "XGBoost - Calendar + Lags",
        calendar_lag_features
    )
)


# ------------------------------------------------
# MODEL 4: LAGS ONLY
# ------------------------------------------------

lag_only_features = [
    feature
    for feature in FEATURES
    if feature.startswith("lag_")
]

lag_only_features += [
    "start_station_id"
]

lag_only_features = [
    feature
    for feature in lag_only_features
    if feature in FEATURES
]

results.append(
    train_and_evaluate(
        "XGBoost - Lags Only",
        lag_only_features
    )
)


# ------------------------------------------------
# MODEL 5: CALENDAR ONLY
# ------------------------------------------------

calendar_only_features = [
    feature
    for feature in calendar_features
    if feature in FEATURES
]

results.append(
    train_and_evaluate(
        "XGBoost - Calendar Only",
        calendar_only_features
    )
)


# ================================================================
# 14. RESULTS
# ================================================================

section("11. ROBUSTNESS COMPARISON")

results_df = pd.DataFrame(results)

print(
    results_df.to_string(index=False)
)

results_path = os.path.join(
    OUTPUT_DIR,
    "robustness_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nSaved: {results_path}"
)


# ================================================================
# 15. R² DIFFERENCE
# ================================================================

section("12. R² ROBUSTNESS ANALYSIS")

full_r2 = results_df.loc[
    results_df["Model"] == "XGBoost - All Features",
    "Test R2"
].iloc[0]

for _, row in results_df.iterrows():

    difference = row["Test R2"] - full_r2

    print(
        f"{row['Model']:<45} "
        f"Test R² = {row['Test R2']:.6f} "
        f"Difference = {difference:+.6f}"
    )


# ================================================================
# 16. FINAL AUDIT SUMMARY
# ================================================================

section("13. FEATURE LEAKAGE & ROBUSTNESS AUDIT SUMMARY")

print(
    f"Total rows                 : {len(df):,}"
)

print(
    f"Total model features      : {len(FEATURES)}"
)

print(
    f"Historical features       : {len(historical_features)}"
)

print(
    f"Potentially risky features : {len(risky_features)}"
)

print(
    f"Original XGBoost Test R²  : {full_r2:.6f}"
)

best_without_risky = results_df.loc[
    results_df["Model"] ==
    "XGBoost - Without Historical Aggregates",
    "Test R2"
].iloc[0]

print(
    f"Without risky aggregates  : "
    f"{best_without_risky:.6f}"
)

print(
    f"R² difference              : "
    f"{best_without_risky - full_r2:+.6f}"
)


# ================================================================
# 17. FINAL MESSAGE
# ================================================================

print("\n" + "=" * 70)
print("FEATURE LEAKAGE & ROBUSTNESS AUDIT COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    "- historical_feature_audit.csv"
)

print(
    "- lag_alignment_test.csv"
)

print(
    "- rolling_alignment_test.csv"
)

print(
    "- aggregate_feature_audit.csv"
)

print(
    "- robustness_comparison.csv"
)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print("\nNext step:")
print(
    "Review robustness_comparison.csv before finalizing the model."
)

print("=" * 70)