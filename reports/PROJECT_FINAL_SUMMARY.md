# URBAN BIKE-SHARING DEMAND PREDICTION
## Comprehensive Project Summary & Final Academic Documentation

---

### 1. Project Title
**Urban Bike-Sharing Demand Prediction Using Machine Learning: Station-Level Hourly Demand Forecasting with Leakage-Controlled Gradient Boosting and Ensemble Methods**

---

### 2. Problem Statement
Urban bike-sharing systems require efficient station-level inventory balancing to avoid stockouts (empty stations preventing checkouts) and overcrowding (full stations preventing returns). Predicting hourly demand accurately across heterogeneous urban stations is challenging due to:
- Complex multi-scale seasonality (hourly commuter peaks, day-of-week patterns, annual weather/seasonal fluctuations).
- High spatial heterogeneity across downtown, business district, transit hub, and residential docks.
- High temporal autocorrelation where immediate historical demand informs future demand.

The goal of this project is to develop, audit, and benchmark machine learning regression models to predict hourly checkout demand ($y_t \ge 0$) at the individual station level, ensuring complete freedom from data leakage, strict chronological evaluation, and zero forward-looking bias.

---

### 3. Dataset Summary
- **Primary Source**: Real-world municipal bike-sharing trip transactions ([trip.csv](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/data/raw/trip.csv)).
- **Aggregated Station-Hour Observations**: **1,230,810 rows**
- **Columns**: 65 (54 final model features, 1 target variable `demand`, temporal metadata `hour_timestamp`, station identifiers).
- **Temporal Span**: August 29, 2013 09:00:00 to August 31, 2015 23:00:00 (733 days / 17,559 unique hourly intervals).
- **Target Variable**: `demand` (integer count of completed bike checkouts per station per hour).
  - Minimum: 0
  - Maximum: 52
  - Mean: 0.5443
  - Standard Deviation: 1.5453
- **Data Quality**: 0 missing values across model features, 0 duplicate `(start_station_id, hour_timestamp)` records.

---

### 4. Preprocessing Pipeline
The raw transaction records were cleaned and transformed into continuous station-hour time series:
1. **Timestamp Parsing**: Ingestion of checkout timestamps using ISO datetime parsing with strict validation (`pd.to_datetime(..., errors="coerce")`).
2. **Station-Hour Grid Expansion**: Generating a complete Cartesian product of all active stations across every operating hour to accurately record zero-demand hours (preventing drop-out bias where zero rentals are omitted).
3. **Special Event Mapping**: Mapping calendar dates across all station-hour records to 9 event categories (`No Event`, `Christmas`, `New Year`, `Halloween`, `Concert`, `Music Festival`, `Food Festival`, `Street Festival`, `Marathon / Cycling Event`).
4. **Chronological Sorting**: Sorting all observations by `[hour_timestamp, start_station_id]`.
5. **Data Type Casting**: Ensuring station IDs, calendar integers, and floating-point continuous features are cast to fixed-precision formats.

---

### 5. Feature Engineering
Feature extraction produced 54 engineered indicators grouped into 6 distinct categories:
1. **Calendar & Temporal Attributes (14 features)**:
   - `start_station_id`, `special_event`, `hour`, `day`, `weekday`, `month`, `year`, `day_of_year`, `week_of_year`, `is_weekend`, `is_weekday`, `morning_peak` (07:00–09:00), `evening_peak` (16:00–19:00), `peak_hour` (union of peak windows).
2. **Cyclical Sinusoidal Encodings (8 features)**:
   - Sine and cosine transformations preserving periodic continuity: `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`, `month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos`.
3. **Temporal Interaction Features (4 features)**:
   - Composite interaction terms: `hour_weekday`, `weekend_hour`, `weekday_hour`, `month_hour`.
4. **Strict Historical Lagged Demand (13 features)**:
   - Short-term momentum: `lag_1h`, `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`.
   - Diurnal and multi-day seasonality: `lag_24h`, `lag_48h`, `lag_72h`, `same_hour_previous_day`, `same_hour_previous_2days`, `same_hour_previous_3days`.
   - Weekly seasonality: `lag_168h` (exact 7-day lag), `same_hour_previous_week`.
5. **Rolling Window Aggregations & Ratios (15 features)**:
   - Moving averages: `rolling_mean_3h`, `rolling_mean_6h`, `rolling_mean_12h`, `rolling_mean_24h`, `rolling_mean_168h`.
   - Moving extrema: `rolling_max_24h`, `rolling_max_168h`, `rolling_min_24h`, `rolling_min_168h`.
   - Moving dispersion & medians: `rolling_std_24h`, `rolling_std_168h`, `rolling_median_24h`.
   - Momentum ratios: `lag_1h_ratio_24h`, `lag_24h_ratio_week`, `rolling_ratio_24h_168h`.

---

### 6. Feature Leakage Audit
A comprehensive audit ([final_feature_audit.py](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/Model%20Evaluation/final_feature_audit.py)) was executed to ensure zero data leakage:
- **Target Isolation**: Target variable `demand` is completely excluded from the predictor matrix $X$.
- **Forward-Looking Terms Elimination**: Excluded all terms containing future, current, or target indicators (`current_demand`, `future_demand`, `target_demand`, `next_hour_demand`).
- **Excluded Non-Causal Aggregations**: Explicitly audited and excluded 9 problematic features that exhibited target leakage or lookahead bias during initial experiments:
  - `demand_change_1h`
  - `demand_change_24h`
  - `demand_change_week`
  - `historical_demand`
  - `historical_max_demand`
  - `historical_min_demand`
  - `station_weekday_hour_mean`
  - `station_hour_historical_mean`
- **Lag Alignment Verification**: Verified that `lag_1h` corresponds strictly to observation $t-1$, `lag_24h` to $t-24$, and `lag_168h` to $t-168$ without any offset error.

---

### 7. Final 54 Leakage-Audited Features Roster
The authoritative feature set is preserved in [final_feature_list.csv](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/data/model_results/leakage_audit/final_feature_list.csv):

| Index | Feature Name | Category | Description |
| :--- | :--- | :--- | :--- |
| 1 | `start_station_id` | Categorical Identifier | Numeric station identifier |
| 2 | `special_event` | Categorical Identifier | Encoded calendar event (0: No Event .. 8: Marathon) |
| 3–7 | `hour`, `day`, `weekday`, `month`, `year` | Calendar | Discrete temporal coordinates |
| 8–9 | `day_of_year`, `week_of_year` | Calendar | Annual time coordinates |
| 10–11 | `is_weekend`, `is_weekday` | Binary Indicator | Workday vs weekend flag |
| 12–14 | `morning_peak`, `evening_peak`, `peak_hour` | Binary Indicator | Commuter peak window indicators |
| 15–22 | `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`, `month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos` | Cyclical Transformations | Trigonometric periodic representations |
| 23–26 | `hour_weekday`, `weekend_hour`, `weekday_hour`, `month_hour` | Interaction Terms | Compound cross-feature encodings |
| 27–39 | `lag_1h`, `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`, `lag_24h`, `lag_48h`, `lag_72h`, `lag_168h`, `same_hour_previous_day`, `same_hour_previous_2days`, `same_hour_previous_3days`, `same_hour_previous_week` | Historical Lags | Previous observed demand values |
| 40–51 | `rolling_mean_3h`, `rolling_mean_6h`, `rolling_mean_12h`, `rolling_mean_24h`, `rolling_mean_168h`, `rolling_max_24h`, `rolling_max_168h`, `rolling_min_24h`, `rolling_min_168h`, `rolling_std_24h`, `rolling_std_168h`, `rolling_median_24h` | Rolling Statistics | Historical moving window aggregations |
| 52–54 | `lag_1h_ratio_24h`, `lag_24h_ratio_week`, `rolling_ratio_24h_168h` | Normalized Ratios | Volatility and trend acceleration ratios |

---

### 8. Train / Validation / Test Methodology
To prevent temporal data leakage and respect time-series dependencies:
- **Splitting Mechanism**: Chronological partition based on **unique timestamps** (not arbitrary random shuffling).
- **Proportions**: 70% Train, 15% Validation, 15% Test.
- **Split Bounds**:
  - **TRAIN (70%)**: 861,560 rows | 2013-08-29 09:00:00 to 2015-01-24 04:00:00
  - **VALIDATION (15%)**: 184,590 rows | 2015-01-24 05:00:00 to 2015-05-14 01:00:00
  - **TEST (15%)**: 184,660 rows | 2015-05-14 02:00:00 to 2015-08-31 23:00:00
- **Overlap Audit**:
  - `Train ∩ Validation`: 0 keys (NONE)
  - `Validation ∩ Test`: 0 keys (NONE)
  - `Train ∩ Test`: 0 keys (NONE)
  - Temporal continuity: `train_max < val_min < val_max < test_min` (PASSED).

---

### 9. Random Forest Regressor
- **Architecture**: Ensembled bagging trees using `RandomForestRegressor` from scikit-learn.
- **Hyperparameters**:
  - `n_estimators`: 200
  - `max_depth`: 15
  - `min_samples_split`: 10
  - `min_samples_leaf`: 2
  - `max_features`: "sqrt"
  - `random_state`: 42
  - `n_jobs`: -1
- **File Artifact**: [random_forest_final_model.pkl](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/models/random_forest_final_model.pkl) (223 MB)
- **Training Time**: 36.42 seconds
- **Prediction Latency**: 1.0055 seconds

---

### 10. XGBoost Regressor
- **Architecture**: Extreme Gradient Boosting using `XGBRegressor` with exact regularized leaf splitting.
- **Hyperparameters**:
  - `n_estimators`: 600
  - `max_depth`: 10
  - `learning_rate`: 0.04
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `min_child_weight`: 1
  - `reg_alpha`: 0
  - `reg_lambda`: 1
  - `objective`: "reg:squarederror"
  - `random_state`: 42
  - `n_jobs`: -1
- **File Artifact**: [xgboost_final_model.pkl](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/models/xgboost_final_model.pkl) (24.4 MB)
- **Training Time**: 24.53 seconds
- **Prediction Latency**: 0.7258 seconds

---

### 11. CatBoost Regressor
- **Architecture**: Symmetric oblivious decision trees with ordered boosting and native categorical handling using `CatBoostRegressor`.
- **Hyperparameters**:
  - `iterations`: 600
  - `depth`: 8
  - `learning_rate`: 0.05
  - `l2_leaf_reg`: 3
  - `loss_function`: "RMSE"
  - `eval_metric`: "RMSE"
  - `random_seed`: 42
  - `thread_count`: -1
- **File Artifact**: [catboost_final_model.pkl](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/models/catboost_final_model.pkl) (4.2 MB)
- **Training Time**: 31.43 seconds
- **Prediction Latency**: 0.0721 seconds (fastest inference)

---

### 12. Measured Evaluation Metrics
All three models were evaluated under identical conditions on the untouched test set (184,660 observations).

| Model | Features | Validation MAE | Validation RMSE | Validation $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Max Error | Median Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | 54 | 0.417276 | 0.883352 | 0.736426 | **0.436126** | **0.897665** | **0.737738** | 22.9567 | 0.1219 |
| **XGBoost** | 54 | 0.413275 | 0.886056 | 0.734809 | **0.430796** | **0.900574** | **0.736036** | 25.8653 | 0.1189 |
| **Random Forest** | 54 | 0.422720 | 0.893354 | 0.730423 | **0.439265** | **0.908459** | **0.731394** | 22.6288 | 0.1261 |

#### Observations:
- **$R^2$ Variance Explained**: CatBoost achieved a test $R^2$ of 0.737738, closely followed by XGBoost at 0.736036 (difference of 0.001702) and Random Forest at 0.731394.
- **Mean Absolute Error (MAE)**: XGBoost attained the lowest test MAE (0.430796), indicating slightly tighter average error on typical low-demand hours.
- **Root Mean Squared Error (RMSE)**: CatBoost attained the lowest test RMSE (0.897665), demonstrating superior penalization of outlier demand surges.
- **Overfitting Resistance**: All models showed tight consistency between validation ($R^2 \approx 0.730 - 0.736$) and test ($R^2 \approx 0.731 - 0.738$) sets.

---

### 13. Error & Residual Analysis
From [final_model_evaluation/](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/data/model_results/final_model_evaluation):
1. **Hourly Error Distribution**: MAE peaks during 08:00–09:00 ($\approx 0.72$) and 17:00–18:00 ($\approx 0.78$) due to elevated commuter demand volatility. Nighttime hours (01:00–05:00) show lowest MAE ($\approx 0.14$).
2. **Day of Week Error**: Weekdays exhibit higher absolute errors ($\text{MAE} \approx 0.45 - 0.47$) than weekends ($\text{MAE} \approx 0.33 - 0.35$), directly proportional to ridership volume.
3. **Monthly Error**: Test set summer months (June–August) show slightly elevated MAE ($\approx 0.44$) corresponding to peak seasonal activity.
4. **Peak vs Non-Peak Error**: Peak hours exhibit an average MAE of 0.702 compared to 0.335 for non-peak hours.
5. **Residual Centering**: Residual distributions for all three models are tightly unimodal and zero-centered with symmetric tails.

---

### 14. Feature Importance Comparison
Comparative importance rankings from [feature_importance_comparison.csv](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/data/model_results/final_model_evaluation/feature_importance_comparison.csv):
- **Top 5 Consensus Features Across All Models**:
  1. `same_hour_previous_week` (primary weekly seasonality anchor)
  2. `lag_168h` (exact 7-day cyclical memory)
  3. `peak_hour` (commuter demand indicator)
  4. `lag_1h` (immediate momentum and station baseline)
  5. `lag_24h` / `same_hour_previous_day` (diurnal repetition)
- *Note*: Feature importance denotes statistical variance reduction within tree splits and does not imply direct physical or economic causality.

---

### 15. Dashboard Architecture & Functionality
Implemented in [Dashboard/app.py](file:///c:/Users/bella/OneDrive/Desktop/Urban%20bike%20sharing%20deman%20prediction/Dashboard/app.py) using Streamlit:
- **Interactive Demand Forecasting**: Allows station selection, date picking, hour selection, and Special Event specification (with calendar auto-suggestion). Validates 54 features against authenticated historical data and alerts user if historical lags are unavailable.
- **Model Comparison Table & Plots**: Renders benchmark comparisons for $R^2$, MAE, and RMSE.
- **Exploratory Data Analysis (EDA)**: Interactive visualizers for hourly, daily, monthly, and station-level demand.
- **Error & Residual Visualizations**: Displays error breakdown tables and embeds all 14 evaluation figures.
- **Feature Importance Viewer**: Dynamic model-specific top-15 feature bar charts and tables.
- **Data Quality & Audit Panel**: Full transparency display of temporal splitting, zero leakage, and zero missing values.

---

### 16. Limitations
1. **Severe Outlier Surges**: Special events, sports matches, and transit strikes can cause abrupt unannounced demand spikes that regression models underestimate.
2. **Absence of Weather Telemetry**: Weather attributes (precipitation, wind speed, temperature) were not present in the historical dataset, limiting response to sudden rain events.
3. **Station Rebalancing Blindness**: Predictions reflect raw user checkout demand; they do not explicitly model bike rebalancing van interventions.

---

### 17. Future Improvements
1. **Weather API Ingestion**: Incorporate NOAA hourly rainfall, temperature, and wind speed data.
2. **Spatial Graph Neural Networks (GNNs)**: Model station network spatial topology using Graph Convolutional Networks (GCNs) or Spatio-Temporal Graph Neural Networks (ST-GNNs).
3. **Automated Redistribution Van Routing**: Connect demand predictions directly to a Vehicle Routing Problem (VRP) solver for real-time truck dispatch.

---

### 18. Final Project Directory Structure

```
Urban bike sharing deman prediction/
├── Dashboard/
│   └── app.py
├── data/
│   ├── eda/
│   │   ├── plots/
│   │   │   ├── correlation_matrix.png
│   │   │   ├── daily_demand.png
│   │   │   ├── hourly_demand.png
│   │   │   ├── monthly_demand.png
│   │   │   ├── top20_stations.png
│   │   │   └── weekday_demand.png
│   │   ├── correlation_matrix.csv
│   │   ├── daily_demand.csv
│   │   ├── duplicate_analysis.csv
│   │   ├── duration_analysis.csv
│   │   ├── eda_quality_summary.csv
│   │   ├── hourly_demand.csv
│   │   ├── missing_values.csv
│   │   ├── monthly_demand.csv
│   │   ├── station_demand.csv
│   │   ├── station_flow_analysis.csv
│   │   ├── statistical_summary.csv
│   │   ├── subscription_demand.csv
│   │   ├── subscription_type.csv
│   │   ├── unique_values.csv
│   │   ├── weekday_demand.csv
│   │   ├── weekday_hour_demand.csv
│   │   └── weekday_vs_weekend.csv
│   ├── features/
│   │   ├── feature_list.csv
│   │   └── station_hourly_demand_features.csv
│   ├── preprocessed/
│   │   └── station_hourly_demand.csv
│   ├── raw/
│   │   └── trip.csv
│   └── model_results/
│       ├── final_model_comparison.csv
│       ├── model_comparison_mae.png
│       ├── model_comparison_r2.png
│       ├── model_comparison_rmse.png
│       ├── catboost/
│       │   ├── catboost_feature_importance.csv
│       │   ├── catboost_metadata.csv
│       │   ├── catboost_results.csv
│       │   └── catboost_test_predictions.csv
│       ├── final_model/
│       │   ├── final_feature_importance.csv
│       │   ├── final_model_metadata.csv
│       │   ├── final_test_predictions.csv
│       │   └── final_xgboost_results.csv
│       ├── final_model_evaluation/
│       │   ├── all_model_test_predictions.csv
│       │   ├── error_by_hour.csv
│       │   ├── error_by_month.csv
│       │   ├── error_by_station.csv
│       │   ├── error_by_weekday.csv
│       │   ├── feature_importance_comparison.csv
│       │   ├── final_evaluation_summary.csv
│       │   ├── final_model_evaluation_report.txt
│       │   ├── peak_vs_nonpeak_error.csv
│       │   ├── top_features_summary.csv
│       │   └── plots/
│       │       ├── actual_vs_predicted_catboost.png
│       │       ├── actual_vs_predicted_random_forest.png
│       │       ├── actual_vs_predicted_xgboost.png
│       │       ├── error_by_hour_all_models.png
│       │       ├── error_by_month_all_models.png
│       │       ├── error_by_weekday_all_models.png
│       │       ├── feature_importance_comparison.png
│       │       ├── model_test_mae_comparison.png
│       │       ├── model_test_r2_comparison.png
│       │       ├── model_test_rmse_comparison.png
│       │       ├── prediction_error_distribution.png
│       │       ├── residual_distribution_catboost.png
│       │       ├── residual_distribution_random_forest.png
│       │       └── residual_distribution_xgboost.png
│       ├── leakage_audit/
│       │   ├── final_audit_summary.csv
│       │   ├── final_feature_list.csv
│       │   ├── final_lag_alignment_test.csv
│       │   └── final_same_hour_alignment_test.csv
│       └── random_forest/
│           ├── random_forest_feature_importance.csv
│           ├── random_forest_metadata.csv
│           ├── random_forest_results.csv
│           └── random_forest_test_predictions.csv
├── EDA/
│   └── eda.py
├── Feature Engineering/
│   └── feature_engineering.py
├── Model Evaluation/
│   ├── compare_final_models.py
│   ├── dashboard_model_compatibility_report.txt
│   ├── evaluate_models.py
│   ├── feature_leakage_audit.py
│   ├── final_feature_audit.py
│   └── final_model_evaluation.py
├── Model Training/
│   ├── train_catboost.py
│   ├── train_final_xgboost.py
│   ├── train_models.py
│   └── train_random_forest.py
├── models/
│   ├── catboost_final_model.pkl
│   ├── random_forest_final_model.pkl
│   └── xgboost_final_model.pkl
├── Preprocessing/
│   └── preprocessing.py
├── Redistribution/
│   └── redistribution.py
└── PROJECT_FINAL_SUMMARY.md
```
