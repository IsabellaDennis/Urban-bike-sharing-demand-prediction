import os
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "station_hourly_demand_features.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

RESULT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "model_results"
)

PLOT_DIR = os.path.join(
    RESULT_DIR,
    "plots"
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("URBAN BIKE-SHARING DEMAND PREDICTION")
print("MODEL EVALUATION AND ERROR ANALYSIS")
print("=" * 70)


# ==============================================================
# 1. LOAD DATASET
# ==============================================================

print("\n1. LOADING FEATURE-ENGINEERED DATASET")
print("-" * 70)

df = pd.read_csv(FEATURE_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ==============================================================
# 2. CONVERT TIMESTAMP
# ==============================================================

print("\n2. CONVERTING TIMESTAMP")
print("-" * 70)

df["hour_timestamp"] = pd.to_datetime(
    df["hour_timestamp"],
    errors="coerce"
)

invalid_timestamps = df["hour_timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps}")

if invalid_timestamps > 0:
    raise ValueError("Invalid timestamps detected.")

print(f"Start: {df['hour_timestamp'].min()}")
print(f"End  : {df['hour_timestamp'].max()}")


# ==============================================================
# 3. SORT DATA
# ==============================================================

df = df.sort_values(
    ["hour_timestamp", "start_station_id"]
).reset_index(drop=True)


# ==============================================================
# 4. TARGET ANALYSIS
# ==============================================================

print("\n3. TARGET ANALYSIS")
print("-" * 70)

TARGET = "demand"

print(f"Target minimum : {df[TARGET].min()}")
print(f"Target maximum : {df[TARGET].max()}")
print(f"Target mean    : {df[TARGET].mean():.4f}")
print(f"Target std     : {df[TARGET].std():.4f}")


# ==============================================================
# 5. DEFINE FEATURES
# ==============================================================

print("\n4. PREPARING FEATURES")
print("-" * 70)

NON_FEATURE_COLUMNS = [
    "demand",
    "hour_timestamp",
    "start_station_name"
]

FEATURES = [
    col for col in df.columns
    if col not in NON_FEATURE_COLUMNS
]

print(f"Number of features: {len(FEATURES)}")


# ==============================================================
# 6. BASIC LEAKAGE CHECK
# ==============================================================

print("\n5. BASIC TARGET LEAKAGE CHECK")
print("-" * 70)

print(
    f"Target demand inside X: "
    f"{'TRUE' if 'demand' in FEATURES else 'FALSE'}"
)

direct_target_features = [
    col for col in FEATURES
    if col.lower() in ["demand", "target"]
]

if direct_target_features:
    print("WARNING: Direct target features found:")
    print(direct_target_features)
else:
    print("No direct target-named features detected.")


# ==============================================================
# 7. CHECK POTENTIAL LEAKAGE FEATURES
# ==============================================================

print("\n6. HISTORICAL FEATURE CHECK")
print("-" * 70)

historical_features = [
    col for col in FEATURES
    if any(
        keyword in col.lower()
        for keyword in [
            "lag",
            "rolling",
            "historical",
            "previous",
            "change",
            "ratio"
        ]
    )
]

print(f"Historical features detected: {len(historical_features)}")

for feature in historical_features:
    print(f"  - {feature}")


# ==============================================================
# 8. CHRONOLOGICAL SPLIT
# ==============================================================

print("\n7. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("-" * 70)

unique_times = np.sort(
    df["hour_timestamp"].unique()
)

n_times = len(unique_times)

train_end_index = int(n_times * 0.70)
validation_end_index = int(n_times * 0.85)

train_end_time = unique_times[train_end_index - 1]
validation_end_time = unique_times[validation_end_index - 1]

train_mask = (
    df["hour_timestamp"] <= train_end_time
)

validation_mask = (
    (df["hour_timestamp"] > train_end_time)
    &
    (df["hour_timestamp"] <= validation_end_time)
)

test_mask = (
    df["hour_timestamp"] > validation_end_time
)

train_df = df[train_mask].copy()
validation_df = df[validation_mask].copy()
test_df = df[test_mask].copy()


def print_split(name, data):
    print(f"\n{name}")
    print(f"Rows : {len(data):,}")
    print(
        f"Time : {data['hour_timestamp'].min()} "
        f"to {data['hour_timestamp'].max()}"
    )


print_split("TRAIN", train_df)
print_split("VALIDATION", validation_df)
print_split("TEST", test_df)


# ==============================================================
# 9. TEMPORAL LEAKAGE CHECK
# ==============================================================

print("\n8. TEMPORAL LEAKAGE CHECK")
print("-" * 70)

if train_df["hour_timestamp"].max() < validation_df["hour_timestamp"].min():
    print("Train -> Validation: PASSED")
else:
    print("Train -> Validation: FAILED")

if validation_df["hour_timestamp"].max() < test_df["hour_timestamp"].min():
    print("Validation -> Test: PASSED")
else:
    print("Validation -> Test: FAILED")


# ==============================================================
# 10. PREPARE X AND y
# ==============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ==============================================================
# 11. TARGET CORRELATION ANALYSIS
# ==============================================================

print("\n9. TARGET CORRELATION ANALYSIS")
print("-" * 70)

numeric_features = df[FEATURES].select_dtypes(
    include=[np.number]
).columns

correlations = (
    df[numeric_features]
    .corrwith(df[TARGET])
    .abs()
    .sort_values(ascending=False)
)

print("\nTop 15 absolute correlations with demand:")

print(
    correlations.head(15).to_string()
)


# ==============================================================
# 12. HIGH CORRELATION WARNING
# ==============================================================

print("\n10. HIGH CORRELATION WARNING")
print("-" * 70)

high_corr = correlations[
    correlations > 0.90
]

if len(high_corr) > 0:

    print(
        "WARNING: Features with absolute correlation > 0.90:"
    )

    print(high_corr)

else:

    print(
        "No feature has absolute correlation > 0.90."
    )


# ==============================================================
# 13. MODEL LOADING FUNCTION
# ==============================================================

print("\n11. LOADING TRAINED MODELS")
print("-" * 70)


def load_model(path):

    # ----------------------------------------------------------
    # Method 1: joblib
    # ----------------------------------------------------------

    try:

        model = joblib.load(path)

        print(
            f"Loaded using joblib: "
            f"{os.path.basename(path)}"
        )

        return model

    except Exception as e:

        print(
            f"joblib failed for "
            f"{os.path.basename(path)}"
        )

        print(
            f"Reason: {type(e).__name__}: {e}"
        )


    # ----------------------------------------------------------
    # Method 2: standard pickle
    # ----------------------------------------------------------

    try:

        with open(path, "rb") as file:

            model = pickle.load(file)

        print(
            f"Loaded using pickle: "
            f"{os.path.basename(path)}"
        )

        return model

    except Exception as e:

        print(
            f"pickle failed for "
            f"{os.path.basename(path)}"
        )

        print(
            f"Reason: {type(e).__name__}: {e}"
        )


    return None


# ==============================================================
# 14. MODEL FILES
# ==============================================================

model_files = {
    "LightGBM": os.path.join(
        MODEL_DIR,
        "lightgbm_model.pkl"
    ),

    "XGBoost": os.path.join(
        MODEL_DIR,
        "xgboost_model.pkl"
    ),

    "Random Forest": os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    ),

    "Extra Trees": os.path.join(
        MODEL_DIR,
        "extra_trees_model.pkl"
    )
}


models = {}

for name, path in model_files.items():

    if not os.path.exists(path):

        print(
            f"{name}: FILE NOT FOUND"
        )

        continue

    model = load_model(path)

    if model is not None:

        models[name] = model


# ==============================================================
# 15. STOP IF MODELS CANNOT BE LOADED
# ==============================================================

if len(models) == 0:

    raise RuntimeError(
        "\nNo trained models could be loaded.\n"
        "The model files need to be retrained."
    )


# ==============================================================
# 16. MODEL EVALUATION
# ==============================================================

print("\n12. MODEL EVALUATION")
print("-" * 70)

results = []

predictions = {}


for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"EVALUATING: {name}")
    print("=" * 70)

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    validation_prediction = model.predict(
        X_validation
    )

    validation_prediction = np.asarray(
        validation_prediction
    ).reshape(-1)

    validation_mae = mean_absolute_error(
        y_validation,
        validation_prediction
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_prediction
        )
    )

    validation_r2 = r2_score(
        y_validation,
        validation_prediction
    )


    # ----------------------------------------------------------
    # Test
    # ----------------------------------------------------------

    test_prediction = model.predict(
        X_test
    )

    test_prediction = np.asarray(
        test_prediction
    ).reshape(-1)

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )


    # ----------------------------------------------------------
    # Store predictions
    # ----------------------------------------------------------

    predictions[name] = test_prediction


    # ----------------------------------------------------------
    # Print
    # ----------------------------------------------------------

    print(
        f"\nValidation MAE  : {validation_mae:.6f}"
    )

    print(
        f"Validation RMSE : {validation_rmse:.6f}"
    )

    print(
        f"Validation R²   : {validation_r2:.6f}"
    )

    print(
        f"\nTest MAE        : {test_mae:.6f}"
    )

    print(
        f"Test RMSE       : {test_rmse:.6f}"
    )

    print(
        f"Test R²         : {test_r2:.6f}"
    )


    results.append({

        "Model": name,

        "Validation MAE":
            validation_mae,

        "Validation RMSE":
            validation_rmse,

        "Validation R2":
            validation_r2,

        "Test MAE":
            test_mae,

        "Test RMSE":
            test_rmse,

        "Test R2":
            test_r2
    })


# ==============================================================
# 17. MODEL COMPARISON
# ==============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Test R2",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ==============================================================
# 18. SAVE MODEL COMPARISON
# ==============================================================

comparison_file = os.path.join(
    RESULT_DIR,
    "evaluation_model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ==============================================================
# 19. SAVE TEST PREDICTIONS
# ==============================================================

print("\n13. SAVING TEST PREDICTIONS")
print("-" * 70)

prediction_df = test_df[
    [
        "start_station_id",
        "start_station_name",
        "hour_timestamp",
        "demand"
    ]
].copy()

for name, prediction in predictions.items():

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    prediction_df[
        f"{safe_name}_prediction"
    ] = prediction


prediction_file = os.path.join(
    RESULT_DIR,
    "test_predictions_all_models.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(
    f"Saved: {prediction_file}"
)


# ==============================================================
# 20. ERROR ANALYSIS
# ==============================================================

print("\n14. ERROR ANALYSIS")
print("-" * 70)

best_model_name = results_df.iloc[0]["Model"]

best_predictions = predictions[
    best_model_name
]

error_df = test_df[
    [
        "start_station_id",
        "start_station_name",
        "hour_timestamp",
        "demand"
    ]
].copy()

error_df["prediction"] = best_predictions

error_df["error"] = (
    error_df["demand"]
    -
    error_df["prediction"]
)

error_df["absolute_error"] = (
    error_df["error"]
    .abs()
)

error_df["squared_error"] = (
    error_df["error"] ** 2
)


# ==============================================================
# 21. ERROR STATISTICS
# ==============================================================

print(
    f"\nBest model for error analysis: "
    f"{best_model_name}"
)

print(
    f"Mean Absolute Error : "
    f"{error_df['absolute_error'].mean():.6f}"
)

print(
    f"Maximum Absolute Error : "
    f"{error_df['absolute_error'].max():.6f}"
)

print(
    f"Mean Error : "
    f"{error_df['error'].mean():.6f}"
)


# ==============================================================
# 22. WORST PREDICTIONS
# ==============================================================

worst_predictions = (
    error_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)

worst_file = os.path.join(
    RESULT_DIR,
    "worst_predictions.csv"
)

worst_predictions.to_csv(
    worst_file,
    index=False
)


print(
    f"\nWorst predictions saved to:"
    f"\n{worst_file}"
)


# ==============================================================
# 23. ACTUAL VS PREDICTED PLOT
# ==============================================================

print("\n15. GENERATING ACTUAL VS PREDICTED PLOT")
print("-" * 70)

plt.figure(
    figsize=(10, 6)
)

sample_size = min(
    5000,
    len(error_df)
)

sample = error_df.head(
    sample_size
)

plt.scatter(
    sample["demand"],
    sample["prediction"],
    alpha=0.3
)

minimum = min(
    sample["demand"].min(),
    sample["prediction"].min()
)

maximum = max(
    sample["demand"].max(),
    sample["prediction"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Demand"
)

plt.ylabel(
    "Predicted Demand"
)

plt.title(
    f"Actual vs Predicted Demand - {best_model_name}"
)

plt.tight_layout()

actual_predicted_plot = os.path.join(
    PLOT_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_plot,
    dpi=300
)

plt.close()


# ==============================================================
# 24. RESIDUAL DISTRIBUTION
# ==============================================================

print("\n16. GENERATING RESIDUAL PLOT")
print("-" * 70)

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    error_df["error"],
    bins=50
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    f"Residual Distribution - {best_model_name}"
)

plt.tight_layout()

residual_plot = os.path.join(
    PLOT_DIR,
    "residual_distribution.png"
)

plt.savefig(
    residual_plot,
    dpi=300
)

plt.close()


# ==============================================================
# 25. DEMAND BY HOUR ERROR ANALYSIS
# ==============================================================

print("\n17. HOURLY ERROR ANALYSIS")
print("-" * 70)

error_df["hour"] = (
    error_df["hour_timestamp"]
    .dt.hour
)

hourly_error = (
    error_df
    .groupby("hour")
    .agg(
        actual_demand=("demand", "mean"),
        predicted_demand=("prediction", "mean"),
        mae=("absolute_error", "mean")
    )
    .reset_index()
)

hourly_error_file = os.path.join(
    RESULT_DIR,
    "hourly_error_analysis.csv"
)

hourly_error.to_csv(
    hourly_error_file,
    index=False
)


# ==============================================================
# 26. STATION ERROR ANALYSIS
# ==============================================================

print("\n18. STATION ERROR ANALYSIS")
print("-" * 70)

station_error = (
    error_df
    .groupby(
        [
            "start_station_id",
            "start_station_name"
        ]
    )
    .agg(
        actual_demand=("demand", "mean"),
        predicted_demand=("prediction", "mean"),
        mae=("absolute_error", "mean"),
        max_error=("absolute_error", "max")
    )
    .reset_index()
    .sort_values(
        "mae",
        ascending=False
    )
)

station_error_file = os.path.join(
    RESULT_DIR,
    "station_error_analysis.csv"
)

station_error.to_csv(
    station_error_file,
    index=False
)


# ==============================================================
# 27. CONFUSION-LIKE DEMAND CATEGORY ANALYSIS
# ==============================================================

print("\n19. DEMAND CATEGORY ANALYSIS")
print("-" * 70)


def demand_category(value):

    if value == 0:
        return "Zero"

    elif value <= 2:
        return "Low"

    elif value <= 5:
        return "Medium"

    else:
        return "High"


error_df["actual_category"] = (
    error_df["demand"]
    .apply(demand_category)
)

error_df["predicted_category"] = (
    error_df["prediction"]
    .clip(lower=0)
    .round()
    .astype(int)
    .apply(demand_category)
)


category_matrix = pd.crosstab(
    error_df["actual_category"],
    error_df["predicted_category"]
)

category_file = os.path.join(
    RESULT_DIR,
    "demand_category_analysis.csv"
)

category_matrix.to_csv(
    category_file
)


# ==============================================================
# 28. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION SUMMARY")
print("=" * 70)

best_row = results_df.iloc[0]

print(
    f"Highest Test R² in this evaluation: "
    f"{best_row['Test R2']:.6f}"
)

print(
    f"Model: {best_row['Model']}"
)

print(
    f"Test MAE : "
    f"{best_row['Test MAE']:.6f}"
)

print(
    f"Test RMSE: "
    f"{best_row['Test RMSE']:.6f}"
)


# ==============================================================
# 29. OUTPUT FILES
# ==============================================================

print("\nGenerated files:")

print(
    f"- {comparison_file}"
)

print(
    f"- {prediction_file}"
)

print(
    f"- {worst_file}"
)

print(
    f"- {hourly_error_file}"
)

print(
    f"- {station_error_file}"
)

print(
    f"- {category_file}"
)

print(
    f"- {actual_predicted_plot}"
)

print(
    f"- {residual_plot}"
)


print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)