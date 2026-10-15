import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ================================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# MODEL TRAINING
# ================================================================

print("=" * 70)
print("URBAN BIKE-SHARING DEMAND PREDICTION")
print("MODEL TRAINING")
print("=" * 70)

# ================================================================
# 1. PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "station_hourly_demand_features.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "data", "model_results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ================================================================
# 2. LOAD DATA
# ================================================================

print("\n1. LOADING FEATURE-ENGINEERED DATASET")
print("-" * 70)

df = pd.read_csv(FEATURE_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df)}")

# ================================================================
# 3. TIMESTAMP CONVERSION
# ================================================================

print("\n2. CONVERTING TIMESTAMP")
print("-" * 70)

df["hour_timestamp"] = pd.to_datetime(
    df["hour_timestamp"],
    errors="coerce"
)

invalid = df["hour_timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid}")

if invalid > 0:
    raise ValueError("Invalid timestamps found.")

df = df.sort_values(
    ["hour_timestamp", "start_station_id"]
).reset_index(drop=True)

print(f"Start: {df['hour_timestamp'].min()}")
print(f"End  : {df['hour_timestamp'].max()}")

# ================================================================
# 4. TARGET
# ================================================================

TARGET = "demand"

if TARGET not in df.columns:
    raise ValueError("Target column 'demand' not found.")

print("\n3. TARGET ANALYSIS")
print("-" * 70)

print(f"Target minimum : {df[TARGET].min()}")
print(f"Target maximum : {df[TARGET].max()}")
print(f"Target mean    : {df[TARGET].mean():.4f}")
print(f"Target std     : {df[TARGET].std():.4f}")

# ================================================================
# 5. REMOVE NON-MODEL COLUMNS
# ================================================================

print("\n4. PREPARING FEATURES")
print("-" * 70)

DROP_COLUMNS = [
    "demand",
    "hour_timestamp",
    "start_station_name"
]

feature_columns = [
    col for col in df.columns
    if col not in DROP_COLUMNS
]

X = df[feature_columns]
y = df[TARGET]

print(f"Number of features: {len(feature_columns)}")

# ================================================================
# 6. CHRONOLOGICAL SPLIT
# ================================================================

print("\n5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("-" * 70)

unique_times = np.sort(
    df["hour_timestamp"].unique()
)

n_times = len(unique_times)

train_time = unique_times[int(n_times * 0.70)]
validation_time = unique_times[int(n_times * 0.85)]

train_mask = df["hour_timestamp"] < train_time

validation_mask = (
    (df["hour_timestamp"] >= train_time) &
    (df["hour_timestamp"] < validation_time)
)

test_mask = df["hour_timestamp"] >= validation_time

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[validation_mask]
y_val = y.loc[validation_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("\nTRAIN")
print(f"Rows : {len(X_train):,}")
print(
    f"Time : {df.loc[train_mask, 'hour_timestamp'].min()} "
    f"to {df.loc[train_mask, 'hour_timestamp'].max()}"
)

print("\nVALIDATION")
print(f"Rows : {len(X_val):,}")
print(
    f"Time : {df.loc[validation_mask, 'hour_timestamp'].min()} "
    f"to {df.loc[validation_mask, 'hour_timestamp'].max()}"
)

print("\nTEST")
print(f"Rows : {len(X_test):,}")
print(
    f"Time : {df.loc[test_mask, 'hour_timestamp'].min()} "
    f"to {df.loc[test_mask, 'hour_timestamp'].max()}"
)

# ================================================================
# 7. VERIFY TEMPORAL ORDER
# ================================================================

print("\n6. TEMPORAL LEAKAGE CHECK")
print("-" * 70)

train_end = df.loc[train_mask, "hour_timestamp"].max()
val_start = df.loc[validation_mask, "hour_timestamp"].min()
val_end = df.loc[validation_mask, "hour_timestamp"].max()
test_start = df.loc[test_mask, "hour_timestamp"].min()

print(f"Train ends      : {train_end}")
print(f"Validation starts: {val_start}")
print(f"Validation ends : {val_end}")
print(f"Test starts     : {test_start}")

if not (
    train_end < val_start < val_end < test_start
):
    raise ValueError("Chronological split validation failed.")

print("Chronological split: PASSED")

# ================================================================
# 8. MODEL DEFINITIONS
# ================================================================

print("\n7. DEFINING MODELS")
print("-" * 70)

models = {

    "LightGBM": LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),

    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
}

# ================================================================
# 9. TRAINING
# ================================================================

results = []
feature_importance_results = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    val_start_time = time.time()

    val_pred = model.predict(X_val)

    val_prediction_time = time.time() - val_start_time

    val_mae = mean_absolute_error(
        y_val,
        val_pred
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_pred
        )
    )

    val_r2 = r2_score(
        y_val,
        val_pred
    )

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    test_start_time = time.time()

    test_pred = model.predict(X_test)

    test_prediction_time = time.time() - test_start_time

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

    print(f"\nValidation MAE  : {val_mae:.4f}")
    print(f"Validation RMSE : {val_rmse:.4f}")
    print(f"Validation R²   : {val_r2:.4f}")

    print(f"\nTest MAE        : {test_mae:.4f}")
    print(f"Test RMSE       : {test_rmse:.4f}")
    print(f"Test R²         : {test_r2:.4f}")

    print(f"\nTraining time   : {training_time:.2f} sec")
    print(f"Prediction time : {test_prediction_time:.4f} sec")

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    model_filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_model.pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"\nModel saved:")
    print(model_path)

    # ------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "feature": feature_columns,
            "importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        )

        feature_importance_results[
            model_name
        ] = importance_df

        importance_file = os.path.join(
            RESULT_DIR,
            f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
        )

        importance_df.to_csv(
            importance_file,
            index=False
        )

    # ------------------------------------------------------------
    # Save test predictions
    # ------------------------------------------------------------

    prediction_df = pd.DataFrame({
        "hour_timestamp": df.loc[test_mask, "hour_timestamp"].values,
        "start_station_id": df.loc[test_mask, "start_station_id"].values,
        "actual_demand": y_test.values,
        "predicted_demand": test_pred
    })

    prediction_file = os.path.join(
        RESULT_DIR,
        f"{model_name.lower().replace(' ', '_')}_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    results.append({
        "Model": model_name,

        "Validation MAE": val_mae,
        "Validation RMSE": val_rmse,
        "Validation R2": val_r2,

        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Test R2": test_r2,

        "Training Time (s)": training_time,
        "Prediction Time (s)": test_prediction_time
    })

# ================================================================
# 10. MODEL COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Test R2",
    ascending=False
)

print("\n")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

comparison_file = os.path.join(
    RESULT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)

# ================================================================
# 11. BEST MODEL BY TEST R²
# ================================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)

print(
    f"\nHighest Test R² in this experiment: "
    f"{best_model['Test R2']:.4f}"
)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"Test MAE : {best_model['Test MAE']:.4f}"
)

print(
    f"Test RMSE: {best_model['Test RMSE']:.4f}"
)

# ================================================================
# 12. SAVE TRAINING METADATA
# ================================================================

metadata = {
    "dataset_rows": len(df),
    "number_features": len(feature_columns),
    "number_stations": df["start_station_id"].nunique(),
    "target": TARGET,
    "target_min": int(y.min()),
    "target_max": int(y.max()),
    "target_mean": float(y.mean()),
    "train_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(X_test),
    "train_end": str(train_end),
    "validation_start": str(val_start),
    "validation_end": str(val_end),
    "test_start": str(test_start),
    "best_model": best_model["Model"],
    "best_test_r2": float(best_model["Test R2"])
}

metadata_df = pd.DataFrame(
    [metadata]
)

metadata_file = os.path.join(
    RESULT_DIR,
    "training_metadata.csv"
)

metadata_df.to_csv(
    metadata_file,
    index=False
)

# ================================================================
# FINAL
# ================================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")
print(f"- {comparison_file}")
print(f"- {metadata_file}")

print("\nModels saved in:")
print(MODEL_DIR)

print("\nModel results saved in:")
print(RESULT_DIR)

print("\nNext step:")
print("Run model evaluation and error analysis.")
print("=" * 70)