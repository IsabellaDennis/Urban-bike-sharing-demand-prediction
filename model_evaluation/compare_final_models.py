# ================================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# FINAL MODEL COMPARISON & REPORTING
# Fair comparison across Random Forest, XGBoost, and CatBoost
# ================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "model_results"
)

RF_RESULT_PATH = os.path.join(
    RESULT_DIR,
    "random_forest",
    "random_forest_results.csv"
)

CB_RESULT_PATH = os.path.join(
    RESULT_DIR,
    "catboost",
    "catboost_results.csv"
)

XGB_RESULT_PATH = os.path.join(
    RESULT_DIR,
    "final_model",
    "final_xgboost_results.csv"
)

COMPARISON_CSV_PATH = os.path.join(
    RESULT_DIR,
    "final_model_comparison.csv"
)

PLOT_R2_PATH = os.path.join(
    RESULT_DIR,
    "model_comparison_r2.png"
)

PLOT_MAE_PATH = os.path.join(
    RESULT_DIR,
    "model_comparison_mae.png"
)

PLOT_RMSE_PATH = os.path.join(
    RESULT_DIR,
    "model_comparison_rmse.png"
)


def load_model_results():
    results = []

    # 1. Random Forest
    if os.path.exists(RF_RESULT_PATH):
        rf_df = pd.read_csv(RF_RESULT_PATH)
        rf_df["Model"] = "Random Forest"
        results.append(rf_df)
    else:
        print(f"Warning: Random Forest results not found at {RF_RESULT_PATH}")

    # 2. XGBoost
    if os.path.exists(XGB_RESULT_PATH):
        xgb_df = pd.read_csv(XGB_RESULT_PATH)
        xgb_df["Model"] = "XGBoost"
        results.append(xgb_df)
    else:
        print(f"Warning: XGBoost results not found at {XGB_RESULT_PATH}")

    # 3. CatBoost
    if os.path.exists(CB_RESULT_PATH):
        cb_df = pd.read_csv(CB_RESULT_PATH)
        cb_df["Model"] = "CatBoost"
        results.append(cb_df)
    else:
        print(f"Warning: CatBoost results not found at {CB_RESULT_PATH}")

    if not results:
        raise FileNotFoundError("No model results found to compare.")

    combined_df = pd.concat(results, ignore_index=True)

    # Standardize column naming if necessary
    rename_dict = {
        "Training Time (s)": "Training Time",
        "Prediction Time (s)": "Prediction Time",
        "Validation R2": "Validation R²",
        "Test R2": "Test R²"
    }
    for old_col, new_col in rename_dict.items():
        if old_col in combined_df.columns and new_col not in combined_df.columns:
            combined_df[new_col] = combined_df[old_col]

    columns_order = [
        "Model",
        "Features",
        "Training Rows",
        "Validation Rows",
        "Test Rows",
        "Validation MAE",
        "Validation RMSE",
        "Validation R²",
        "Test MAE",
        "Test RMSE",
        "Test R²",
        "Training Time",
        "Prediction Time"
    ]

    available_columns = [c for c in columns_order if c in combined_df.columns]
    combined_df = combined_df[available_columns]

    return combined_df


def generate_plots(df: pd.DataFrame):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    models = df["Model"].tolist()
    colors = ["#2b5c8f", "#e07a5f", "#3d8a55"][:len(models)]

    # 1. R² Comparison Plot (Validation vs Test)
    plt.figure(figsize=(10, 6))
    x = range(len(models))
    width = 0.35

    val_r2 = df["Validation R²"] if "Validation R²" in df.columns else df["Validation R2"]
    test_r2 = df["Test R²"] if "Test R²" in df.columns else df["Test R2"]

    plt.bar([i - width/2 for i in x], val_r2, width=width, label="Validation R²", color="#457b9d")
    plt.bar([i + width/2 for i in x], test_r2, width=width, label="Test R²", color="#1d3557")

    for i in x:
        plt.text(i - width/2, val_r2.iloc[i] + 0.01, f"{val_r2.iloc[i]:.4f}", ha="center", va="bottom", fontsize=10)
        plt.text(i + width/2, test_r2.iloc[i] + 0.01, f"{test_r2.iloc[i]:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xlabel("Model", fontweight="bold")
    plt.ylabel("R² Score (Higher is better)", fontweight="bold")
    plt.title("Model Comparison: R² Score (Validation vs Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, models)
    plt.ylim(0, max(max(val_r2), max(test_r2)) * 1.15)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOT_R2_PATH, dpi=300)
    plt.close()
    print(f"R² comparison plot saved: {PLOT_R2_PATH}")

    # 2. MAE Comparison Plot (Validation vs Test)
    plt.figure(figsize=(10, 6))
    val_mae = df["Validation MAE"]
    test_mae = df["Test MAE"]

    plt.bar([i - width/2 for i in x], val_mae, width=width, label="Validation MAE", color="#e9c46a")
    plt.bar([i + width/2 for i in x], test_mae, width=width, label="Test MAE", color="#e76f51")

    for i in x:
        plt.text(i - width/2, val_mae.iloc[i] + 0.005, f"{val_mae.iloc[i]:.4f}", ha="center", va="bottom", fontsize=10)
        plt.text(i + width/2, test_mae.iloc[i] + 0.005, f"{test_mae.iloc[i]:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xlabel("Model", fontweight="bold")
    plt.ylabel("Mean Absolute Error (Lower is better)", fontweight="bold")
    plt.title("Model Comparison: MAE (Validation vs Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, models)
    plt.ylim(0, max(max(val_mae), max(test_mae)) * 1.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_MAE_PATH, dpi=300)
    plt.close()
    print(f"MAE comparison plot saved: {PLOT_MAE_PATH}")

    # 3. RMSE Comparison Plot (Validation vs Test)
    plt.figure(figsize=(10, 6))
    val_rmse = df["Validation RMSE"]
    test_rmse = df["Test RMSE"]

    plt.bar([i - width/2 for i in x], val_rmse, width=width, label="Validation RMSE", color="#6a4c93")
    plt.bar([i + width/2 for i in x], test_rmse, width=width, label="Test RMSE", color="#1982c4")

    for i in x:
        plt.text(i - width/2, val_rmse.iloc[i] + 0.01, f"{val_rmse.iloc[i]:.4f}", ha="center", va="bottom", fontsize=10)
        plt.text(i + width/2, test_rmse.iloc[i] + 0.01, f"{test_rmse.iloc[i]:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xlabel("Model", fontweight="bold")
    plt.ylabel("Root Mean Squared Error (Lower is better)", fontweight="bold")
    plt.title("Model Comparison: RMSE (Validation vs Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, models)
    plt.ylim(0, max(max(val_rmse), max(test_rmse)) * 1.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_RMSE_PATH, dpi=300)
    plt.close()
    print(f"RMSE comparison plot saved: {PLOT_RMSE_PATH}")


def print_final_report(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    for _, row in df.iterrows():
        model = row["Model"]
        print(f"\n{model}")
        print(f"Features: {int(row['Features'])}")
        print(f"Validation MAE:  {row['Validation MAE']:.6f}")
        print(f"Validation RMSE: {row['Validation RMSE']:.6f}")
        val_r2 = row["Validation R²"] if "Validation R²" in row else row["Validation R2"]
        test_r2 = row["Test R²"] if "Test R²" in row else row["Test R2"]
        print(f"Validation R2:   {val_r2:.6f}")
        print(f"Test MAE:        {row['Test MAE']:.6f}")
        print(f"Test RMSE:       {row['Test RMSE']:.6f}")
        print(f"Test R2:         {test_r2:.6f}")

    print("\n" + "=" * 60)


def main():
    print("=" * 70)
    print("URBAN BIKE-SHARING DEMAND PREDICTION")
    print("FINAL MODEL EVALUATION AND COMPARISON")
    print("=" * 70)

    df = load_model_results()

    # Save final comparison CSV
    df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"\nFinal model comparison saved to:\n{COMPARISON_CSV_PATH}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False, justify="center"))

    # Generate plots
    generate_plots(df)

    # Print report
    print_final_report(df)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
