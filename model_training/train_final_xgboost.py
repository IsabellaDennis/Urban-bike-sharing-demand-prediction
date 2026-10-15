# ================================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# FINAL XGBOOST MODEL TRAINING
# Using 53 leakage-audited features
# ================================================================

import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "station_hourly_demand_features.csv"
)

FEATURE_LIST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "model_results",
    "leakage_audit",
    "final_feature_list.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "model_results",
    "final_model"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_final_model.pkl"
)

RESULT_PATH = os.path.join(
    RESULT_DIR,
    "final_xgboost_results.csv"
)

PREDICTION_PATH = os.path.join(
    RESULT_DIR,
    "final_test_predictions.csv"
)

IMPORTANCE_PATH = os.path.join(
    RESULT_DIR,
    "final_feature_importance.csv"
)

METADATA_PATH = os.path.join(
    RESULT_DIR,
    "final_model_metadata.csv"
)


TARGET_COLUMN = "demand"
TIMESTAMP_COLUMN = "hour_timestamp"
STATION_COLUMN = "start_station_id"

EXPECTED_FEATURE_COUNT = 54


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("URBAN BIKE-SHARING DEMAND PREDICTION")
print("FINAL XGBOOST MODEL TRAINING")
print("=" * 70)


# ================================================================
# 1. LOAD DATASET
# ================================================================

print("\n1. LOADING FEATURE-ENGINEERED DATASET")
print("-" * 70)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ================================================================
# 2. CHECK REQUIRED COLUMNS
# ================================================================

print("\n2. CHECKING REQUIRED COLUMNS")
print("-" * 70)

required_columns = [
    TIMESTAMP_COLUMN,
    TARGET_COLUMN,
    STATION_COLUMN
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

print("All required columns are available.")


# ================================================================
# 3. CONVERT TIMESTAMP
# ================================================================

print("\n3. CONVERTING TIMESTAMP")
print("-" * 70)

df[TIMESTAMP_COLUMN] = pd.to_datetime(
    df[TIMESTAMP_COLUMN],
    errors="coerce"
)

invalid_timestamps = df[TIMESTAMP_COLUMN].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps}")

if invalid_timestamps > 0:
    raise ValueError(
        "Invalid timestamps detected."
    )

print(
    f"Start: {df[TIMESTAMP_COLUMN].min()}"
)

print(
    f"End  : {df[TIMESTAMP_COLUMN].max()}"
)


# ================================================================
# 4. SORT DATA
# ================================================================

print("\n4. SORTING DATA")
print("-" * 70)

df = df.sort_values(
    [TIMESTAMP_COLUMN, STATION_COLUMN]
).reset_index(drop=True)

print(
    "Data sorted chronologically by timestamp and station."
)


# ================================================================
# 5. TARGET ANALYSIS
# ================================================================

print("\n5. TARGET ANALYSIS")
print("-" * 70)

target = df[TARGET_COLUMN]

print(
    f"Target minimum : {target.min():.0f}"
)

print(
    f"Target maximum : {target.max():.0f}"
)

print(
    f"Target mean    : {target.mean():.4f}"
)

print(
    f"Target std     : {target.std():.4f}"
)


# ================================================================
# 6. LOAD FINAL AUDITED FEATURE LIST
# ================================================================

print("\n6. LOADING FINAL AUDITED FEATURE LIST")
print("-" * 70)

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

possible_feature_columns = [
    "Feature",
    "feature",
    "features",
    "column",
    "feature_name"
]

feature_column = None

for col in possible_feature_columns:

    if col in feature_df.columns:
        feature_column = col
        break

if feature_column is None:

    if len(feature_df.columns) == 1:

        feature_column = feature_df.columns[0]

    else:

        raise ValueError(
            "Could not identify feature column in "
            "final_feature_list.csv"
        )


features = (
    feature_df[feature_column]
    .dropna()
    .astype(str)
    .tolist()
)

print(
    f"Features loaded from audit: {len(features)}"
)


# ================================================================
# 7. VERIFY FINAL FEATURE COUNT
# ================================================================

print("\n7. VERIFYING FINAL FEATURE SET")
print("-" * 70)

if len(features) != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} features, "
        f"but found {len(features)}."
    )

print(
    f"Final audited feature count: {len(features)}"
)

print("\nFinal features:")

for i, feature in enumerate(features, start=1):

    print(
        f"{i:2d}. {feature}"
    )


# ================================================================
# 8. TARGET LEAKAGE CHECK
# ================================================================

print("\n8. TARGET LEAKAGE CHECK")
print("-" * 70)

if TARGET_COLUMN in features:

    raise ValueError(
        "TARGET LEAKAGE DETECTED: demand is inside X."
    )

print(
    "Target demand inside feature list: FALSE"
)

print(
    "No direct target leakage detected."
)


# ================================================================
# 9. CURRENT / FUTURE DEMAND CHECK
# ================================================================

print("\n9. CURRENT / FUTURE DEMAND FEATURE CHECK")
print("-" * 70)

dangerous_terms = [
    "current_demand",
    "future_demand",
    "target_demand",
    "next_hour_demand",
    "future",
    "current"
]

risky_names = []

for feature in features:

    feature_lower = feature.lower()

    for term in dangerous_terms:

        if term in feature_lower:

            risky_names.append(feature)
            break


if risky_names:

    print(
        "Potential current/future demand features:"
    )

    for feature in risky_names:

        print(f" - {feature}")

    raise ValueError(
        "Potential current/future demand leakage detected."
    )

else:

    print(
        "No current/future-demand feature names detected."
    )


# ================================================================
# 10. FEATURE AVAILABILITY CHECK
# ================================================================

print("\n10. CHECKING FEATURE AVAILABILITY")
print("-" * 70)

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print("Missing features:")

    for feature in missing_features:

        print(f" - {feature}")

    raise ValueError(
        "Some final audited features are missing."
    )

print(
    "All final audited features are available."
)


# ================================================================
# 11. MISSING VALUE CHECK
# ================================================================

print("\n11. CHECKING MISSING VALUES")
print("-" * 70)

missing_values = (
    df[features]
    .isna()
    .sum()
    .sum()
)

print(
    f"Total missing feature values: {missing_values}"
)

if missing_values > 0:

    raise ValueError(
        "Missing values detected in model features."
    )

print(
    "No missing feature values."
)


# ================================================================
# 12. DUPLICATE STATION-HOUR CHECK
# ================================================================

print("\n12. CHECKING STATION-HOUR DUPLICATES")
print("-" * 70)

duplicate_count = df.duplicated(
    subset=[
        STATION_COLUMN,
        TIMESTAMP_COLUMN
    ]
).sum()

print(
    f"Duplicate station-hour records: {duplicate_count}"
)

if duplicate_count > 0:

    raise ValueError(
        "Duplicate station-hour records detected."
    )

print(
    "No duplicate station-hour records."
)


# ================================================================
# 13. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ================================================================

print("\n13. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("-" * 70)

# ------------------------------------------------
# IMPORTANT:
#
# Split using UNIQUE TIMESTAMPS rather than rows.
#
# This ensures that a complete timestamp belongs
# to exactly one dataset split.
#
# Approximate:
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# ------------------------------------------------

unique_times = np.sort(
    df[TIMESTAMP_COLUMN].unique()
)

number_of_times = len(unique_times)

train_time_end_index = int(
    number_of_times * 0.70
)

validation_time_end_index = int(
    number_of_times * 0.85
)

train_end_time = unique_times[
    train_time_end_index - 1
]

validation_end_time = unique_times[
    validation_time_end_index - 1
]

validation_start_time = unique_times[
    train_time_end_index
]

test_start_time = unique_times[
    validation_time_end_index
]

train_df = df[
    df[TIMESTAMP_COLUMN] <= train_end_time
].copy()

validation_df = df[
    (df[TIMESTAMP_COLUMN] >= validation_start_time)
    &
    (df[TIMESTAMP_COLUMN] <= validation_end_time)
].copy()

test_df = df[
    df[TIMESTAMP_COLUMN] >= test_start_time
].copy()


print("\nTRAIN")

print(
    f"Rows : {len(train_df):,}"
)

print(
    f"Time : "
    f"{train_df[TIMESTAMP_COLUMN].min()} "
    f"to "
    f"{train_df[TIMESTAMP_COLUMN].max()}"
)


print("\nVALIDATION")

print(
    f"Rows : {len(validation_df):,}"
)

print(
    f"Time : "
    f"{validation_df[TIMESTAMP_COLUMN].min()} "
    f"to "
    f"{validation_df[TIMESTAMP_COLUMN].max()}"
)


print("\nTEST")

print(
    f"Rows : {len(test_df):,}"
)

print(
    f"Time : "
    f"{test_df[TIMESTAMP_COLUMN].min()} "
    f"to "
    f"{test_df[TIMESTAMP_COLUMN].max()}"
)


# ================================================================
# 14. TEMPORAL LEAKAGE CHECK
# ================================================================

print("\n14. TEMPORAL LEAKAGE CHECK")
print("-" * 70)

train_time_min = train_df[TIMESTAMP_COLUMN].min()
train_time_max = train_df[TIMESTAMP_COLUMN].max()

validation_time_min = (
    validation_df[TIMESTAMP_COLUMN].min()
)

validation_time_max = (
    validation_df[TIMESTAMP_COLUMN].max()
)

test_time_min = test_df[TIMESTAMP_COLUMN].min()
test_time_max = test_df[TIMESTAMP_COLUMN].max()


print(
    f"Train ends       : {train_time_max}"
)

print(
    f"Validation starts: {validation_time_min}"
)

print(
    f"Validation ends  : {validation_time_max}"
)

print(
    f"Test starts      : {test_time_min}"
)

print(
    f"Test ends        : {test_time_max}"
)


# ------------------------------------------------
# Check chronological ordering
# ------------------------------------------------

if train_time_max >= validation_time_min:

    raise ValueError(
        "Train/validation temporal overlap detected."
    )

if validation_time_max >= test_time_min:

    raise ValueError(
        "Validation/test temporal overlap detected."
    )

if train_time_max >= test_time_min:

    raise ValueError(
        "Train/test temporal overlap detected."
    )


# ------------------------------------------------
# Check exact station-hour row overlap
# ------------------------------------------------

train_keys = set(
    zip(
        train_df[STATION_COLUMN],
        train_df[TIMESTAMP_COLUMN]
    )
)

validation_keys = set(
    zip(
        validation_df[STATION_COLUMN],
        validation_df[TIMESTAMP_COLUMN]
    )
)

test_keys = set(
    zip(
        test_df[STATION_COLUMN],
        test_df[TIMESTAMP_COLUMN]
    )
)


train_validation_overlap = (
    train_keys.intersection(validation_keys)
)

validation_test_overlap = (
    validation_keys.intersection(test_keys)
)

train_test_overlap = (
    train_keys.intersection(test_keys)
)


if train_validation_overlap:

    raise ValueError(
        "Train/validation station-hour overlap detected."
    )

if validation_test_overlap:

    raise ValueError(
        "Validation/test station-hour overlap detected."
    )

if train_test_overlap:

    raise ValueError(
        "Train/test station-hour overlap detected."
    )


print("\nTrain -> Validation: PASSED")
print("Validation -> Test: PASSED")
print("Train -> Test: PASSED")
print("Station-hour overlap: NONE")


# ================================================================
# 15. PREPARE X AND y
# ================================================================

print("\n15. PREPARING MODEL DATA")
print("-" * 70)

X_train = train_df[features].copy()
y_train = train_df[TARGET_COLUMN].copy()

X_validation = validation_df[features].copy()
y_validation = validation_df[TARGET_COLUMN].copy()

X_test = test_df[features].copy()
y_test = test_df[TARGET_COLUMN].copy()


print(
    f"X_train shape     : {X_train.shape}"
)

print(
    f"y_train shape     : {y_train.shape}"
)

print(
    f"X_validation shape: {X_validation.shape}"
)

print(
    f"y_validation shape: {y_validation.shape}"
)

print(
    f"X_test shape      : {X_test.shape}"
)

print(
    f"y_test shape      : {y_test.shape}"
)


# ================================================================
# 16. DEFINE FINAL XGBOOST MODEL
# ================================================================

print("\n16. DEFINING FINAL XGBOOST MODEL")
print("-" * 70)

model_parameters = {

    "n_estimators": 600,

    "max_depth": 10,

    "learning_rate": 0.04,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "min_child_weight": 1,

    "reg_alpha": 0,

    "reg_lambda": 1,

    "objective": "reg:squarederror",

    "eval_metric": "rmse",

    "random_state": 42,

    "n_jobs": -1
}


for key, value in model_parameters.items():

    print(
        f"{key:20s}: {value}"
    )


# ================================================================
# 17. TRAIN MODEL
# ================================================================

print("\n17. TRAINING FINAL XGBOOST")
print("=" * 70)

model = XGBRegressor(
    **model_parameters
)

training_start = time.time()

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_validation, y_validation)
    ],
    verbose=False
)

training_time = (
    time.time() -
    training_start
)

print(
    f"Training completed in "
    f"{training_time:.2f} seconds."
)


# ================================================================
# 18. PREDICTIONS
# ================================================================

print("\n18. GENERATING PREDICTIONS")
print("-" * 70)

prediction_start = time.time()

validation_predictions = model.predict(
    X_validation
)

test_predictions = model.predict(
    X_test
)

prediction_time = (
    time.time() -
    prediction_start
)

print(
    "Predictions generated successfully."
)


# ================================================================
# 19. VALIDATION EVALUATION
# ================================================================

print("\n19. VALIDATION EVALUATION")
print("-" * 70)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)

validation_r2 = r2_score(
    y_validation,
    validation_predictions
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


# ================================================================
# 20. TEST EVALUATION
# ================================================================

print("\n20. TEST EVALUATION")
print("-" * 70)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)


print(
    f"Test MAE        : {test_mae:.6f}"
)

print(
    f"Test RMSE       : {test_rmse:.6f}"
)

print(
    f"Test R²         : {test_r2:.6f}"
)

print(
    f"\nTraining time   : "
    f"{training_time:.2f} sec"
)

print(
    f"Prediction time : "
    f"{prediction_time:.4f} sec"
)


# ================================================================
# 21. SAVE MODEL
# ================================================================

print("\n21. SAVING FINAL MODEL")
print("-" * 70)

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

print(
    "Final model saved:"
)

print(
    MODEL_PATH
)


# ================================================================
# 22. SAVE TEST PREDICTIONS
# ================================================================

print("\n22. SAVING TEST PREDICTIONS")
print("-" * 70)

prediction_df = pd.DataFrame({

    "hour_timestamp":
        test_df[TIMESTAMP_COLUMN].values,

    "start_station_id":
        test_df[STATION_COLUMN].values,

    "actual_demand":
        y_test.values,

    "predicted_demand":
        test_predictions,

    "absolute_error":
        np.abs(
            y_test.values -
            test_predictions
        ),

    "error":
        y_test.values -
        test_predictions

})


prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)

print(
    "Test predictions saved:"
)

print(
    PREDICTION_PATH
)


# ================================================================
# 23. FEATURE IMPORTANCE
# ================================================================

print("\n23. FEATURE IMPORTANCE")
print("-" * 70)

importance_df = pd.DataFrame({

    "Feature":
        features,

    "Importance":
        model.feature_importances_

})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nTop 20 features:")

print(
    importance_df
    .head(20)
    .to_string(index=False)
)


importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False
)

print(
    "\nFeature importance saved:"
)

print(
    IMPORTANCE_PATH
)


# ================================================================
# 24. SAVE RESULTS
# ================================================================

print("\n24. SAVING MODEL RESULTS")
print("-" * 70)

results_df = pd.DataFrame({

    "Model": [
        "Final XGBoost"
    ],

    "Features": [
        len(features)
    ],

    "Training Rows": [
        len(X_train)
    ],

    "Validation Rows": [
        len(X_validation)
    ],

    "Test Rows": [
        len(X_test)
    ],

    "Validation MAE": [
        validation_mae
    ],

    "Validation RMSE": [
        validation_rmse
    ],

    "Validation R2": [
        validation_r2
    ],

    "Test MAE": [
        test_mae
    ],

    "Test RMSE": [
        test_rmse
    ],

    "Test R2": [
        test_r2
    ],

    "Training Time (s)": [
        training_time
    ],

    "Prediction Time (s)": [
        prediction_time
    ]

})


results_df.to_csv(
    RESULT_PATH,
    index=False
)

print(
    "Results saved:"
)

print(
    RESULT_PATH
)


# ================================================================
# 25. SAVE METADATA
# ================================================================

print("\n25. SAVING TRAINING METADATA")
print("-" * 70)

metadata = {

    "dataset_rows":
        len(df),

    "dataset_columns":
        df.shape[1],

    "original_audited_features":
        61,

    "risky_aggregates_removed":
        5,

    "demand_change_features_removed":
        3,

    "final_features":
        len(features),

    "training_rows":
        len(X_train),

    "validation_rows":
        len(X_validation),

    "test_rows":
        len(X_test),

    "train_start":
        str(train_time_min),

    "train_end":
        str(train_time_max),

    "validation_start":
        str(validation_time_min),

    "validation_end":
        str(validation_time_max),

    "test_start":
        str(test_time_min),

    "test_end":
        str(test_time_max),

    "validation_mae":
        validation_mae,

    "validation_rmse":
        validation_rmse,

    "validation_r2":
        validation_r2,

    "test_mae":
        test_mae,

    "test_rmse":
        test_rmse,

    "test_r2":
        test_r2,

    "training_time_seconds":
        training_time,

    "prediction_time_seconds":
        prediction_time,

    "temporal_split":
        "PASSED",

    "station_hour_overlap":
        "NONE",

    "target_leakage":
        "NOT DETECTED",

    "current_future_leakage":
        "NOT DETECTED",

    "missing_values":
        0,

    "feature_count":
        len(features)

}


metadata_df = pd.DataFrame(
    [metadata]
)

metadata_df.to_csv(
    METADATA_PATH,
    index=False
)

print(
    "Metadata saved:"
)

print(
    METADATA_PATH
)


# ================================================================
# 26. FINAL SUMMARY
# ================================================================

print("\n")

print("=" * 70)
print("FINAL XGBOOST MODEL SUMMARY")
print("=" * 70)

print(
    f"Dataset rows        : {len(df):,}"
)

print(
    f"Dataset columns     : {df.shape[1]}"
)

print(
    f"Final model features: {len(features)}"
)

print()

print(
    f"Training rows       : {len(X_train):,}"
)

print(
    f"Validation rows     : {len(X_validation):,}"
)

print(
    f"Test rows           : {len(X_test):,}"
)

print()

print("MODEL PERFORMANCE")
print("-" * 70)

print(
    f"Validation MAE     : "
    f"{validation_mae:.6f}"
)

print(
    f"Validation RMSE    : "
    f"{validation_rmse:.6f}"
)

print(
    f"Validation R²      : "
    f"{validation_r2:.6f}"
)

print()

print(
    f"Test MAE           : "
    f"{test_mae:.6f}"
)

print(
    f"Test RMSE          : "
    f"{test_rmse:.6f}"
)

print(
    f"Test R²            : "
    f"{test_r2:.6f}"
)

print()

print("MODEL CONFIGURATION")
print("-" * 70)

for key, value in model_parameters.items():

    print(
        f"{key:20s}: {value}"
    )

print()

print("DATA QUALITY")
print("-" * 70)

print(
    "Temporal split       : PASSED"
)

print(
    "Station-hour overlap: NONE"
)

print(
    "Target leakage       : NOT DETECTED"
)

print(
    "Current/future leak  : NOT DETECTED"
)

print(
    "Missing values       : 0"
)

print()

print("GENERATED FILES")
print("-" * 70)

print(
    MODEL_PATH
)

print(
    PREDICTION_PATH
)

print(
    IMPORTANCE_PATH
)

print(
    RESULT_PATH
)

print(
    METADATA_PATH
)

print()

print("=" * 70)
print("FINAL XGBOOST TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)