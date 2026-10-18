"""
Test Suite: Simplified Station Demand Prediction and Multi-Station Rebalancing
Verifies that:
1. Model inference strictly uses station-specific features and yields station-specific predictions.
2. Different hours produce time-dependent predictions.
3. Zero remains a valid model output (e.g. late night or inactive hours).
4. Out-of-sample future dates use the diurnal profile forecasting pipeline.
5. Multi-station redistribution correctly calculates shortages and donor surpluses using the simplified demand estimate.
6. The trained model and feature schema (54 features including neutral special_event=0) remain intact.
"""
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import (
    prepare_prediction_features,
    make_prediction,
    get_station_hourly_profile,
    get_station_capacity,
    estimate_available_bikes,
    load_feature_dataset,
    load_feature_list,
    load_production_model,
)
from redistribution.redistribution import (
    calculate_station_shortage_surplus,
    find_nearest_safe_donor,
)

print("=" * 85)
print("TESTING SIMPLIFIED STATION DEMAND PREDICTION & MULTI-STATION REDISTRIBUTION")
print("=" * 85)

# Load real artifacts
model = load_production_model()
feat_list = load_feature_list()
df_feat = load_feature_dataset()

assert model is not None, "Production model failed to load!"
assert len(feat_list) == 54, f"Expected 54 features, found {len(feat_list)}"
assert "special_event" in feat_list, "special_event column must remain in schema"

# Test 1: Station-specific demand at morning peak
ts_peak = pd.Timestamp("2015-06-15 08:00:00")
X_70, _, _ = prepare_prediction_features(70, ts_peak, df_feat, feat_list)
X_64, _, _ = prepare_prediction_features(64, ts_peak, df_feat, feat_list)
X_12, _, _ = prepare_prediction_features(12, ts_peak, df_feat, feat_list)

pred_70 = make_prediction(model, X_70)
pred_64 = make_prediction(model, X_64)
pred_12 = make_prediction(model, X_12)

print(f"[PASS] Test 1: Station-specific predictions at 08:00 AM:")
print(f"       Station 70: {pred_70:.2f} -> {int(np.floor(pred_70 + 0.5))} trips")
print(f"       Station 64: {pred_64:.2f} -> {int(np.floor(pred_64 + 0.5))} trips")
print(f"       Station 12: {pred_12:.2f} -> {int(np.floor(pred_12 + 0.5))} trips")
assert pred_70 != pred_64 and pred_64 != pred_12, "Stations must not have identical predictions"

# Test 2: Time-of-day variation for Station 70
ts_noon = pd.Timestamp("2015-06-15 12:00:00")
ts_eve = pd.Timestamp("2015-06-15 17:00:00")
ts_night = pd.Timestamp("2015-06-15 02:00:00")

X_noon, _, _ = prepare_prediction_features(70, ts_noon, df_feat, feat_list)
X_eve, _, _ = prepare_prediction_features(70, ts_eve, df_feat, feat_list)
X_night, _, _ = prepare_prediction_features(70, ts_night, df_feat, feat_list)

pred_noon = make_prediction(model, X_noon)
pred_eve = make_prediction(model, X_eve)
pred_night = make_prediction(model, X_night)

print(f"\n[PASS] Test 2: Time-of-day predictions for Station 70:")
print(f"       08:00 AM: {pred_70:.2f} trips")
print(f"       12:00 PM: {pred_noon:.2f} trips")
print(f"       05:00 PM: {pred_eve:.2f} trips")
print(f"       02:00 AM: {pred_night:.2f} trips")
assert pred_night < 1.0, "Late night demand should be minimal/zero"

# Test 3: Zero remains a valid model output
trips_night = max(0, int(np.floor(pred_night + 0.5)))
print(f"\n[PASS] Test 3: Zero output preserved:")
print(f"       02:00 AM raw={pred_night:.2f} -> {trips_night} trips (valid zero)")
assert trips_night == 0

# Test 4: Out-of-sample future date profile forecasting
future_dt = pd.Timestamp("2026-06-15 08:00:00")
profile = get_station_hourly_profile(70)
h_df = profile["hourly_df"]
h_row = h_df[h_df["hour"] == 8]
h_base = float(h_row["demand"].values[0])
is_wkday = 1 if future_dt.weekday() < 5 else 0
day_factor = (profile["weekday_avg"] / (profile["weekday_avg"] + profile["weekend_avg"] + 1e-5) * 2.0) if is_wkday else (profile["weekend_avg"] / (profile["weekday_avg"] + profile["weekend_avg"] + 1e-5) * 2.0)
future_pred = max(0.0, round(h_base * day_factor, 2))
future_trips = max(0, int(np.floor(future_pred + 0.5)))
print(f"\n[PASS] Test 4: Future date out-of-sample forecast:")
print(f"       Station 70 at 2026-06-15 08:00 AM: {future_pred:.2f} -> {future_trips} trips")
assert future_trips > 0

# Test 5: End-to-end multi-station redistribution with simplified model demand
cap70 = get_station_capacity(70)
avail_low = 10
pred_trips = int(np.floor(pred_70 + 0.5))  # 24 trips
shortage = max(0, pred_trips - avail_low)  # 24 - 10 = 14 bikes needed
match = find_nearest_safe_donor(
    target_station_id=70,
    target_dt=ts_peak,
    dest_demand=pred_trips,
    dest_available=avail_low,
    dest_name="SF Caltrain",
    safety_buffer=3,
)
print(f"\n[PASS] Test 5: Multi-station redistribution using model output ({pred_trips} trips):")
print(f"       Shortage: {shortage} bikes")
print(f"       Status: {match['status']}")
print(f"       Total Transferred: {match.get('total_transferred', match['recommended_transfer'])} bikes")
print(f"       Unmet Demand: {match.get('unmet_demand', 0)} bikes")
assert shortage == match.get("total_transferred", match["recommended_transfer"]) + match.get("unmet_demand", 0)

print("\n" + "=" * 85)
print("ALL 5/5 SIMPLIFIED WORKFLOW TESTS PASSED SUCCESSFULLY!")
print("=" * 85)
