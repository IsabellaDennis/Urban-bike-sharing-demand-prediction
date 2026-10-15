# ================================================================
# URBAN BIKE-SHARING DEMAND PREDICTION
# FINAL REPRODUCIBLE PROJECT AUDIT
# ================================================================

import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
FEATURES_PATH = DATA_DIR / "features" / "station_hourly_demand_features.csv"
FEATURE_LIST_PATH = DATA_DIR / "model_results" / "leakage_audit" / "final_feature_list.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_RESULTS_DIR = DATA_DIR / "model_results"
EVAL_DIR = MODEL_RESULTS_DIR / "final_model_evaluation"
EVAL_PLOTS_DIR = EVAL_DIR / "plots"

AUDIT_REPORT_PATH = MODEL_RESULTS_DIR / "final_project_audit_report.txt"

EXPECTED_METRICS = {
    "XGBoost": {"MAE": 0.430796, "RMSE": 0.900574, "R2": 0.736036},
    "Random Forest": {"MAE": 0.439265, "RMSE": 0.908459, "R2": 0.731394},
    "CatBoost": {"MAE": 0.436126, "RMSE": 0.897665, "R2": 0.737738}
}

EXPECTED_PLOTS = [
    "model_test_r2_comparison.png",
    "model_test_mae_comparison.png",
    "model_test_rmse_comparison.png",
    "actual_vs_predicted_xgboost.png",
    "actual_vs_predicted_random_forest.png",
    "actual_vs_predicted_catboost.png",
    "residual_distribution_xgboost.png",
    "residual_distribution_random_forest.png",
    "residual_distribution_catboost.png",
    "error_by_hour_all_models.png",
    "error_by_weekday_all_models.png",
    "error_by_month_all_models.png",
    "prediction_error_distribution.png",
    "feature_importance_comparison.png"
]


def main():
    print("=" * 70)
    print("URBAN BIKE-SHARING DEMAND PREDICTION")
    print("FINAL INDEPENDENT PROJECT AUDIT")
    print("=" * 70)

    audit_results = {}
    audit_notes = {}

    # ------------------------------------------------------------
    # CHECK 1 — PROJECT FILES
    # ------------------------------------------------------------
    print("\n[CHECK 1] Verifying Required Project Files...")
    required_files = [
        BASE_DIR / "PROJECT_FINAL_SUMMARY.md",
        BASE_DIR / "Dashboard" / "app.py",
        BASE_DIR / "EDA" / "eda.py",
        BASE_DIR / "Feature Engineering" / "feature_engineering.py",
        BASE_DIR / "Preprocessing" / "preprocessing.py",
        BASE_DIR / "Model Training" / "train_catboost.py",
        BASE_DIR / "Model Training" / "train_random_forest.py",
        BASE_DIR / "Model Training" / "train_final_xgboost.py",
        BASE_DIR / "Model Evaluation" / "compare_final_models.py",
        BASE_DIR / "Model Evaluation" / "final_model_evaluation.py",
        BASE_DIR / "Model Evaluation" / "feature_leakage_audit.py",
        BASE_DIR / "Model Evaluation" / "final_feature_audit.py",
        MODELS_DIR / "xgboost_final_model.pkl",
        MODELS_DIR / "random_forest_final_model.pkl",
        MODELS_DIR / "catboost_final_model.pkl",
        FEATURES_PATH,
        FEATURE_LIST_PATH,
        MODEL_RESULTS_DIR / "final_model" / "final_xgboost_results.csv",
        MODEL_RESULTS_DIR / "random_forest" / "random_forest_results.csv",
        MODEL_RESULTS_DIR / "catboost" / "catboost_results.csv",
        MODEL_RESULTS_DIR / "final_model_comparison.csv"
    ]

    missing_files = [str(f.relative_to(BASE_DIR)) for f in required_files if not f.exists()]
    if missing_files:
        audit_results["Required files"] = "FAIL"
        audit_notes["Required files"] = f"Missing files: {missing_files}"
        print(f"FAILED: Missing {len(missing_files)} required files: {missing_files}")
    else:
        audit_results["Required files"] = "PASS"
        audit_notes["Required files"] = f"All {len(required_files)} required files exist."
        print(f"PASSED: All {len(required_files)} required files exist.")

    # ------------------------------------------------------------
    # CHECK 2 — FEATURE COUNT & UNICITY
    # ------------------------------------------------------------
    print("\n[CHECK 2] Verifying Authoritative 54 Features...")
    feature_df = pd.read_csv(FEATURE_LIST_PATH)
    col = [c for c in ["Feature", "feature", "features", "column"] if c in feature_df.columns][0]
    features = feature_df[col].dropna().astype(str).tolist()

    has_duplicates = len(features) != len(set(features))
    has_event = "special_event" in features
    if len(features) == 54 and not has_duplicates and has_event:
        audit_results["54 feature verification"] = "PASS"
        audit_notes["54 feature verification"] = "Exactly 54 unique features loaded, including special_event."
        print(f"PASSED: Exactly 54 unique features loaded. (Duplicates: {has_duplicates})")
    else:
        audit_results["54 feature verification"] = "FAIL"
        audit_notes["54 feature verification"] = f"Count={len(features)}, Duplicates={has_duplicates}, SpecialEvent={has_event}"
        print(f"FAILED: Count={len(features)}, Duplicates={has_duplicates}")

    print("Complete Feature Roster:")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")

    # ------------------------------------------------------------
    # CHECK 3 — FEATURE EXISTENCE IN DATASET
    # ------------------------------------------------------------
    print("\n[CHECK 3] Verifying Feature Existence in Dataset...")
    header_df = pd.read_csv(FEATURES_PATH, nrows=1)
    missing_in_dataset = [f for f in features if f not in header_df.columns]
    if not missing_in_dataset:
        audit_results["Feature existence"] = "PASS"
        audit_notes["Feature existence"] = "All 54 features exist in station_hourly_demand_features.csv."
        print("PASSED: All 54 features exist in station_hourly_demand_features.csv.")
    else:
        audit_results["Feature existence"] = "FAIL"
        audit_notes["Feature existence"] = f"Missing features: {missing_in_dataset}"
        print(f"FAILED: Missing in dataset: {missing_in_dataset}")

    # ------------------------------------------------------------
    # CHECK 4 — TARGET ISOLATION & LOOKAHEAD LEAKAGE
    # ------------------------------------------------------------
    print("\n[CHECK 4] Verifying Target Isolation & Lookahead Leakage...")
    target_in_features = "demand" in features
    suspicious_terms = ["future", "current", "target", "next_hour"]
    suspicious_found = []
    for f in features:
        for term in suspicious_terms:
            if term in f.lower():
                suspicious_found.append(f)

    if not target_in_features and not suspicious_found:
        audit_results["Target isolation"] = "PASS"
        audit_notes["Target isolation"] = "Zero target/lookahead leakage detected in feature list."
        print("PASSED: Target 'demand' excluded. Zero lookahead terms found.")
    else:
        audit_results["Target isolation"] = "FAIL"
        audit_notes["Target isolation"] = f"Target in features: {target_in_features}, Suspicious: {suspicious_found}"
        print(f"FAILED: Target in features: {target_in_features}, Suspicious: {suspicious_found}")

    # ------------------------------------------------------------
    # LOAD DATASET FOR DATA INTEGRITY CHECKS (CHECKS 5, 6, 7, 8)
    # ------------------------------------------------------------
    print("\nLoading dataset for data integrity checks...")
    cols_to_load = ["start_station_id", "hour_timestamp", "demand"] + features
    # Remove duplicate columns if any
    cols_to_load = list(dict.fromkeys(cols_to_load))
    df = pd.read_csv(FEATURES_PATH, usecols=cols_to_load)
    df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"], errors="coerce")
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns.")

    # ------------------------------------------------------------
    # CHECK 5 — MISSING AND INFINITE VALUES
    # ------------------------------------------------------------
    print("\n[CHECK 5] Checking Missing and Infinite Values...")
    nan_count = int(df[features].isna().sum().sum())
    numeric_features = df[features].select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_features.values).sum())

    if nan_count == 0:
        audit_results["Missing values"] = "PASS"
        audit_notes["Missing values"] = "0 missing values across all 54 features."
        print("PASSED: 0 missing values across all 54 features.")
    else:
        audit_results["Missing values"] = "FAIL"
        audit_notes["Missing values"] = f"{nan_count} missing values found."
        print(f"FAILED: {nan_count} missing values found.")

    if inf_count == 0:
        audit_results["Infinite values"] = "PASS"
        audit_notes["Infinite values"] = "0 infinite values across all 54 features."
        print("PASSED: 0 infinite values across all 54 features.")
    else:
        audit_results["Infinite values"] = "FAIL"
        audit_notes["Infinite values"] = f"{inf_count} infinite values found."
        print(f"FAILED: {inf_count} infinite values found.")

    # ------------------------------------------------------------
    # CHECK 6 — DUPLICATE STATION-HOUR
    # ------------------------------------------------------------
    print("\n[CHECK 6] Checking Duplicate Station-Hour Records...")
    dup_count = int(df.duplicated(subset=["start_station_id", "hour_timestamp"]).sum())
    if dup_count == 0:
        audit_results["Duplicate station-hour records"] = "PASS"
        audit_notes["Duplicate station-hour records"] = "0 duplicate station-hour records."
        print("PASSED: 0 duplicate station-hour records.")
    else:
        audit_results["Duplicate station-hour records"] = "FAIL"
        audit_notes["Duplicate station-hour records"] = f"{dup_count} duplicates found."
        print(f"FAILED: {dup_count} duplicates found.")

    # ------------------------------------------------------------
    # CHECK 7 — CHRONOLOGICAL SPLIT
    # ------------------------------------------------------------
    print("\n[CHECK 7] Recreating Chronological Train/Validation/Test Split...")
    unique_times = np.sort(df["hour_timestamp"].unique())
    n_times = len(unique_times)
    train_idx = int(n_times * 0.70)
    val_idx = int(n_times * 0.85)

    train_end = unique_times[train_idx - 1]
    val_start = unique_times[train_idx]
    val_end = unique_times[val_idx - 1]
    test_start = unique_times[val_idx]

    train_df = df[df["hour_timestamp"] <= train_end]
    val_df = df[(df["hour_timestamp"] >= val_start) & (df["hour_timestamp"] <= val_end)]
    test_df = df[df["hour_timestamp"] >= test_start]

    split_ordered = (train_end < val_start) and (val_end < test_start)

    print(f"TRAIN      : {len(train_df):,} rows ({train_df['hour_timestamp'].min()} to {train_end})")
    print(f"VALIDATION : {len(val_df):,} rows ({val_start} to {val_end})")
    print(f"TEST       : {len(test_df):,} rows ({test_start} to {test_df['hour_timestamp'].max()})")

    if split_ordered and len(train_df) == 861560 and len(val_df) == 184590 and len(test_df) == 184660:
        audit_results["Chronological split"] = "PASS"
        audit_notes["Chronological split"] = "Exact 861,560 / 184,590 / 184,660 split verified with strict chronological ordering."
        print("PASSED: Exact row counts and chronological ordering verified.")
    else:
        audit_results["Chronological split"] = "FAIL"
        audit_notes["Chronological split"] = f"Ordered={split_ordered}, Rows: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        print(f"FAILED: Ordered={split_ordered}, Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # ------------------------------------------------------------
    # CHECK 8 — STATION-HOUR OVERLAP
    # ------------------------------------------------------------
    print("\n[CHECK 8] Checking Station-Hour Key Overlap Across Splits...")
    train_keys = set(zip(train_df["start_station_id"], train_df["hour_timestamp"]))
    val_keys = set(zip(val_df["start_station_id"], val_df["hour_timestamp"]))
    test_keys = set(zip(test_df["start_station_id"], test_df["hour_timestamp"]))

    train_val_overlap = len(train_keys.intersection(val_keys))
    val_test_overlap = len(val_keys.intersection(test_keys))
    train_test_overlap = len(train_keys.intersection(test_keys))

    total_overlap = train_val_overlap + val_test_overlap + train_test_overlap
    if total_overlap == 0:
        audit_results["Station-hour overlap"] = "PASS"
        audit_notes["Station-hour overlap"] = "Zero key overlap across all data partitions."
        print("PASSED: Zero key overlap across Train, Validation, and Test sets.")
    else:
        audit_results["Station-hour overlap"] = "FAIL"
        audit_notes["Station-hour overlap"] = f"Overlap detected: Tr-Val={train_val_overlap}, Val-Te={val_test_overlap}, Tr-Te={train_test_overlap}"
        print(f"FAILED: Overlap detected: {total_overlap}")

    # ------------------------------------------------------------
    # CHECK 9 — MODEL LOADING
    # ------------------------------------------------------------
    print("\n[CHECK 9] Loading Trained Model Checkpoints...")
    loaded_models = {}
    for m_name, p in [
        ("XGBoost", MODELS_DIR / "xgboost_final_model.pkl"),
        ("Random Forest", MODELS_DIR / "random_forest_final_model.pkl"),
        ("CatBoost", MODELS_DIR / "catboost_final_model.pkl")
    ]:
        try:
            with open(p, "rb") as f:
                loaded_models[m_name] = pickle.load(f)
            audit_results[f"{m_name} loading"] = "PASS"
            audit_notes[f"{m_name} loading"] = f"Successfully loaded from {p.name}"
            print(f"PASSED: {m_name} loaded successfully.")
        except Exception as e:
            audit_results[f"{m_name} loading"] = "FAIL"
            audit_notes[f"{m_name} loading"] = f"Error: {e}"
            print(f"FAILED: {m_name} failed to load: {e}")

    # ------------------------------------------------------------
    # CHECK 10 — MODEL FEATURE COMPATIBILITY
    # ------------------------------------------------------------
    print("\n[CHECK 10] Checking Model Feature Compatibility...")
    compat_ok = True
    compat_details = []

    # Prepare a test sample
    sample_row = test_df[features].head(1).copy()

    for m_name, model in loaded_models.items():
        try:
            # Check feature names if exposed
            if hasattr(model, "feature_names_in_"):
                m_features = list(model.feature_names_in_)
                if m_features == features:
                    compat_details.append(f"{m_name}: 54/54 features matched in exact order.")
                else:
                    compat_ok = False
                    compat_details.append(f"{m_name}: Feature mismatch.")
            elif hasattr(model, "feature_names_"):
                m_features = list(model.feature_names_)
                if m_features == features:
                    compat_details.append(f"{m_name}: 54/54 features matched in exact order.")
                else:
                    compat_ok = False
                    compat_details.append(f"{m_name}: Feature mismatch.")
            else:
                # Test prediction shape
                pred = model.predict(sample_row)
                compat_details.append(f"{m_name}: Successfully predicted on 54 columns (model does not expose feature_names attribute directly).")
        except Exception as e:
            compat_ok = False
            compat_details.append(f"{m_name}: Compatibility error: {e}")

    if compat_ok:
        audit_results["Model feature compatibility"] = "PASS"
        audit_notes["Model feature compatibility"] = "; ".join(compat_details)
        print(f"PASSED: All models compatible with 54 features. {compat_details}")
    else:
        audit_results["Model feature compatibility"] = "FAIL"
        audit_notes["Model feature compatibility"] = "; ".join(compat_details)
        print(f"FAILED: {compat_details}")

    # ------------------------------------------------------------
    # CHECK 11 — PREDICTION SANITY TEST
    # ------------------------------------------------------------
    print("\n[CHECK 11] Performing Prediction Sanity Test...")
    sanity_ok = True
    sanity_notes = []

    for m_name, model in loaded_models.items():
        try:
            raw_pred = model.predict(sample_row)[0]
            is_num = isinstance(raw_pred, (int, float, np.floating))
            is_fin = np.isfinite(raw_pred)
            not_nan = not np.isnan(raw_pred)

            clipped_pred = max(float(raw_pred), 0.0)

            if is_num and is_fin and not_nan and clipped_pred >= 0:
                sanity_notes.append(f"{m_name}: pred={raw_pred:.4f} (clipped={clipped_pred:.4f})")
            else:
                sanity_ok = False
                sanity_notes.append(f"{m_name}: Sanity check failed (pred={raw_pred})")
        except Exception as e:
            sanity_ok = False
            sanity_notes.append(f"{m_name}: Error during prediction: {e}")

    if sanity_ok:
        audit_results["Prediction sanity"] = "PASS"
        audit_notes["Prediction sanity"] = "; ".join(sanity_notes)
        print(f"PASSED: Prediction sanity verified. {sanity_notes}")
    else:
        audit_results["Prediction sanity"] = "FAIL"
        audit_notes["Prediction sanity"] = "; ".join(sanity_notes)
        print(f"FAILED: {sanity_notes}")

    # ------------------------------------------------------------
    # CHECK 12 — RESULT FILE CONSISTENCY
    # ------------------------------------------------------------
    print("\n[CHECK 12] Checking Documented vs Saved Result Metrics...")
    consistency_ok = True
    comp_csv = pd.read_csv(MODEL_RESULTS_DIR / "final_model_comparison.csv")

    for m_name, exp in EXPECTED_METRICS.items():
        row = comp_csv[comp_csv["Model"].str.lower() == m_name.lower()]
        if row.empty:
            consistency_ok = False
            print(f"FAILED: Model {m_name} not found in final_model_comparison.csv")
            continue

        r2_col = "Test R²" if "Test R²" in row.columns else "Test R2"
        actual_mae = float(row["Test MAE"].values[0])
        actual_rmse = float(row["Test RMSE"].values[0])
        actual_r2 = float(row[r2_col].values[0])

        mae_diff = abs(actual_mae - exp["MAE"])
        rmse_diff = abs(actual_rmse - exp["RMSE"])
        r2_diff = abs(actual_r2 - exp["R2"])

        if mae_diff < 1e-4 and rmse_diff < 1e-4 and r2_diff < 1e-4:
            print(f"PASSED: {m_name} test metrics matched (MAE: {actual_mae:.6f}, RMSE: {actual_rmse:.6f}, R2: {actual_r2:.6f})")
        else:
            consistency_ok = False
            print(f"FAILED: {m_name} metrics mismatch: MAE diff={mae_diff:.6f}, RMSE diff={rmse_diff:.6f}, R2 diff={r2_diff:.6f}")

    if consistency_ok:
        audit_results["Result consistency"] = "PASS"
        audit_notes["Result consistency"] = "All reported test metrics match documented benchmark values."
    else:
        audit_results["Result consistency"] = "FAIL"
        audit_notes["Result consistency"] = "Mismatch found in documented vs saved metrics."

    # ------------------------------------------------------------
    # CHECK 13 — EVALUATION ARTIFACTS
    # ------------------------------------------------------------
    print("\n[CHECK 13] Checking Evaluation Artifacts and Plots...")
    eval_csvs = [
        EVAL_DIR / "final_evaluation_summary.csv",
        EVAL_DIR / "all_model_test_predictions.csv",
        EVAL_DIR / "error_by_hour.csv",
        EVAL_DIR / "error_by_weekday.csv",
        EVAL_DIR / "error_by_month.csv",
        EVAL_DIR / "error_by_station.csv",
        EVAL_DIR / "peak_vs_nonpeak_error.csv",
        EVAL_DIR / "error_by_special_event.csv",
        EVAL_DIR / "feature_importance_comparison.csv",
        EVAL_DIR / "top_features_summary.csv",
        EVAL_DIR / "final_model_evaluation_report.txt"
    ]

    missing_eval_csvs = [f.name for f in eval_csvs if not f.exists()]
    missing_plots = [p for p in EXPECTED_PLOTS if not (EVAL_PLOTS_DIR / p).exists()]

    if not missing_eval_csvs and not missing_plots:
        audit_results["Evaluation artifacts"] = "PASS"
        audit_notes["Evaluation artifacts"] = f"All {len(eval_csvs)} evaluation CSV/report files and {len(EXPECTED_PLOTS)} plots verified."
        print(f"PASSED: All {len(eval_csvs)} evaluation CSVs and {len(EXPECTED_PLOTS)} PNG plots exist.")
    else:
        audit_results["Evaluation artifacts"] = "FAIL"
        audit_notes["Evaluation artifacts"] = f"Missing CSVs: {missing_eval_csvs}, Missing plots: {missing_plots}"
        print(f"FAILED: Missing CSVs: {missing_eval_csvs}, Missing plots: {missing_plots}")

    # ------------------------------------------------------------
    # CHECK 14 — DASHBOARD COMPATIBILITY & RELATIVE PATHS
    # ------------------------------------------------------------
    print("\n[CHECK 14] Inspecting Dashboard Implementation...")
    app_path = BASE_DIR / "Dashboard" / "app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        app_code = f.read()

    uses_relative = "Path(__file__).resolve()" in app_code and "C:/Users/bella" not in app_code and "C:\\Users\\bella" not in app_code
    uses_54 = "EXPECTED_FEATURE_COUNT = 54" in app_code
    has_event_selector = "Special Event" in app_code and "sb_event" in app_code
    handles_negative = "max(float(pred_raw), 0.0)" in app_code or "max(prediction, 0.0)" in app_code or "max(float(pred), 0.0)" in app_code
    safety_message = "Prediction cannot be generated because the required historical features are unavailable for this timestamp." in app_code

    if uses_relative and uses_54 and has_event_selector and handles_negative and safety_message:
        audit_results["Dashboard compatibility"] = "PASS"
        audit_notes["Dashboard compatibility"] = "Project-relative paths verified, 54 features validated, Special Event input active, non-negative clipping active, fallback safety active."
        print("PASSED: Dashboard implementation complies with all specifications.")
    else:
        audit_results["Dashboard compatibility"] = "FAIL"
        audit_notes["Dashboard compatibility"] = f"Relative={uses_relative}, 54Features={uses_54}, EventSelector={has_event_selector}, NonNegative={handles_negative}, SafetyMsg={safety_message}"
        print(f"FAILED: Relative={uses_relative}, 54Features={uses_54}, EventSelector={has_event_selector}, NonNegative={handles_negative}, SafetyMsg={safety_message}")

    # ------------------------------------------------------------
    # CHECK 15 — OBSOLETE FILE CLEANUP
    # ------------------------------------------------------------
    print("\n[CHECK 15] Verifying Obsolete Files are Absent...")
    obsolete_items = [
        BASE_DIR / "Model Training" / "train_moderate_xgboost.py",
        BASE_DIR / "Model Training" / "tune_xgboost.py",
        BASE_DIR / "Hyperparameter Tuning",
        MODEL_RESULTS_DIR / "moderate_model",
        MODEL_RESULTS_DIR / "tuning"
    ]

    existing_obsolete = [str(p.relative_to(BASE_DIR)) for p in obsolete_items if p.exists()]
    if not existing_obsolete:
        audit_results["Obsolete file cleanup"] = "PASS"
        audit_notes["Obsolete file cleanup"] = "All obsolete experimental scripts and result folders are absent."
        print("PASSED: All obsolete experimental items are absent.")
    else:
        audit_results["Obsolete file cleanup"] = "FAIL"
        audit_notes["Obsolete file cleanup"] = f"Found obsolete items: {existing_obsolete}"
        print(f"FAILED: Found obsolete items: {existing_obsolete}")

    # ------------------------------------------------------------
    # COMPILE & WRITE FINAL AUDIT REPORT
    # ------------------------------------------------------------
    total_checks = len(audit_results)
    passed_checks = sum(1 for status in audit_results.values() if status == "PASS")
    failed_checks = total_checks - passed_checks

    report_lines = [
        "=" * 60,
        "FINAL PROJECT AUDIT",
        "=" * 60,
        "",
        "Project:",
        "Urban Bike-Sharing Demand Prediction",
        "",
        "Dataset:",
        "1,230,810 rows",
        "",
        "Features:",
        "54",
        "",
        "Models:",
        "XGBoost",
        "Random Forest",
        "CatBoost",
        "",
        "Checks:",
        ""
    ]

    for check_name, status in audit_results.items():
        report_lines.append(f"[{status}] {check_name}")

    report_lines.extend([
        "",
        f"TOTAL CHECKS: {total_checks}",
        f"PASSED:       {passed_checks}",
        f"FAILED:       {failed_checks}",
        "",
        "=" * 60,
        "AUDIT DETAILS & NOTES",
        "=" * 60
    ])

    for check_name, note in audit_notes.items():
        report_lines.append(f"- {check_name}: {note}")

    report_lines.extend([
        "",
        "=" * 60,
        "CONFIRMATION: ZERO MODELS WERE RETRAINED DURING THIS AUDIT.",
        "=" * 60
    ])

    report_text = "\n".join(report_lines)
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + "=" * 60)
    print("FINAL AUDIT SUMMARY")
    print("=" * 60)
    print(f"TOTAL CHECKS : {total_checks}")
    print(f"PASSED       : {passed_checks}")
    print(f"FAILED       : {failed_checks}")
    print(f"\nAudit report written to: {AUDIT_REPORT_PATH}")
    print("=" * 60)

    if failed_checks > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
