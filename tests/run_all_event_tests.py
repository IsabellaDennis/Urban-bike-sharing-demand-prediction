"""
SmartBike AI — Station-Specific Demand Prediction and Event Logic Test Suite
Validates station-specific & time-specific model inference, date-sensitive holiday rules,
hour-independent concert/other event scenarios, zero-baseline minimums, and redistribution consistency.
"""
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import (
    calculate_defensible_event_prediction,
    check_date_compatibility,
    prepare_prediction_features,
    make_prediction,
    get_station_hourly_profile,
    get_station_capacity,
    estimate_available_bikes,
    load_feature_dataset,
    load_feature_list,
    load_production_model,
    FIXED_HOLIDAYS,
    EVENT_OPTIONS,
)
from redistribution.redistribution import (
    calculate_station_shortage_surplus,
    find_nearest_safe_donor,
)

print("=" * 85)
print("SMARTBIKE AI — STATION-SPECIFIC DEMAND PREDICTION & EVENT LOGIC TEST SUITE")
print("=" * 85)

test_results = []

def record_test(test_id, name, passed, details):
    status_str = "PASS" if passed else "FAIL"
    test_results.append((test_id, name, status_str, details))
    print(f"[{status_str}] Test {test_id}: {name}")
    print(f"       Details: {details}\n")
    assert passed, f"Test {test_id} FAILED: {details}"

# Load real artifacts
model = load_production_model()
feat_list = load_feature_list()
df_feat = load_feature_dataset()

# --------------------------------------------------------------------------
# Test 1: Three different stations at the same hour
# --------------------------------------------------------------------------
# Weekday Monday 2015-06-15 at 08:00 AM
ts_peak = pd.Timestamp("2015-06-15 08:00:00")
X_70, act_70, _ = prepare_prediction_features(70, ts_peak, df_feat, feat_list, "No Event")
X_64, act_64, _ = prepare_prediction_features(64, ts_peak, df_feat, feat_list, "No Event")
X_12, act_12, _ = prepare_prediction_features(12, ts_peak, df_feat, feat_list, "No Event")

pred_70 = make_prediction(model, X_70)
pred_64 = make_prediction(model, X_64)
pred_12 = make_prediction(model, X_12)

trips_70 = int(np.floor(pred_70 + 0.5))
trips_64 = int(np.floor(pred_64 + 0.5))
trips_12 = int(np.floor(pred_12 + 0.5))

t1_pass = (trips_70 == 24 and trips_64 == 3 and trips_12 == 0 and pred_12 > 0.0)
record_test(
    1,
    "Three different stations at same hour (08:00 AM) — station-specific predictions",
    t1_pass,
    f"Station 70: raw={pred_70:.2f} -> {trips_70} trips | Station 64: raw={pred_64:.2f} -> {trips_64} trips | Station 12: raw={pred_12:.2f} -> {trips_12} trips"
)

# --------------------------------------------------------------------------
# Test 2: One station at three different hours
# --------------------------------------------------------------------------
# Station 70 at 08:00 AM, 12:00 PM, 05:00 PM, and 02:00 AM
ts_noon = pd.Timestamp("2015-06-15 12:00:00")
ts_eve = pd.Timestamp("2015-06-15 17:00:00")
ts_night = pd.Timestamp("2015-06-15 02:00:00")

X_noon, _, _ = prepare_prediction_features(70, ts_noon, df_feat, feat_list, "No Event")
X_eve, _, _ = prepare_prediction_features(70, ts_eve, df_feat, feat_list, "No Event")
X_night, _, _ = prepare_prediction_features(70, ts_night, df_feat, feat_list, "No Event")

pred_noon = make_prediction(model, X_noon)
pred_eve = make_prediction(model, X_eve)
pred_night = make_prediction(model, X_night)

trips_noon = int(np.floor(pred_noon + 0.5))
trips_eve = int(np.floor(pred_eve + 0.5))
trips_night = int(np.floor(pred_night + 0.5))

t2_pass = (trips_70 == 24 and trips_noon == 1 and trips_eve == 11 and trips_night == 0)
record_test(
    2,
    "One station across multiple hours — time-dependent model predictions",
    t2_pass,
    f"08:00 AM={trips_70} (raw {pred_70:.2f}) | 12:00 PM={trips_noon} (raw {pred_noon:.2f}) | 05:00 PM={trips_eve} (raw {pred_eve:.2f}) | 02:00 AM={trips_night} (raw {pred_night:.2f})"
)

# --------------------------------------------------------------------------
# Test 3: Multiple dates and weekdays
# --------------------------------------------------------------------------
# Weekday Monday 2015-06-15 vs Weekend Sunday 2015-06-14 vs Future Date 2026-06-15
ts_sun = pd.Timestamp("2015-06-14 08:00:00")
X_sun, _, _ = prepare_prediction_features(70, ts_sun, df_feat, feat_list, "No Event")
pred_sun = make_prediction(model, X_sun)
trips_sun = int(np.floor(pred_sun + 0.5))

# Out of sample future date
prof_70 = get_station_hourly_profile(70)
h_base_fut = float(prof_70["hourly_df"][prof_70["hourly_df"]["hour"] == 8]["demand"].values[0])
trips_fut = int(np.floor(h_base_fut + 0.5))

t3_pass = (trips_70 == 24 and trips_sun == 0 and pred_sun < 0.5 and trips_fut == 16)
record_test(
    3,
    "Multiple dates & weekdays — weekday commute vs weekend off-peak vs future profile",
    t3_pass,
    f"Weekday 08:00={trips_70} trips | Sunday 08:00={trips_sun} trips (raw {pred_sun:.2f}) | Future 2026 08:00={trips_fut} trips"
)

# --------------------------------------------------------------------------
# Test 4: No Event with positive and zero baselines
# --------------------------------------------------------------------------
flt4_pos, trips4_pos, meta4_pos = calculate_defensible_event_prediction(
    baseline_pred=pred_70, event_name="No Event", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70
)
flt4_zero, trips4_zero, meta4_zero = calculate_defensible_event_prediction(
    baseline_pred=pred_12, event_name="No Event", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=12
)

t4_pass = (trips4_pos == 24 and meta4_pos["adjustment_trips"] == 0 and trips4_zero == 0 and meta4_zero["adjustment_trips"] == 0)
record_test(
    4,
    "No Event — zero adjustment for both positive and zero baselines",
    t4_pass,
    f"Pos: base {pred_70:.2f} -> {trips4_pos} (adj 0) | Zero: base {pred_12:.2f} -> {trips4_zero} (adj 0)"
)

# --------------------------------------------------------------------------
# Test 5: Christmas on December 25 vs invalid date
# --------------------------------------------------------------------------
flt5_valid, trips5_valid, meta5_valid = calculate_defensible_event_prediction(
    baseline_pred=24.0, event_name="Christmas", target_date=datetime.date(2014, 12, 25), target_hour=8, station_id=70
)
flt5_invalid, trips5_invalid, meta5_invalid = calculate_defensible_event_prediction(
    baseline_pred=24.0, event_name="Christmas", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70
)

t5_pass = (
    trips5_valid == 6 and meta5_valid["adjustment_trips"] == -18 and meta5_valid["method_label"] == "Historical Holiday Pattern"
    and trips5_invalid == 24 and meta5_invalid["adjustment_trips"] == 0 and meta5_invalid["method_label"] == "Baseline (Date Mismatch)"
)
record_test(
    5,
    "Christmas — holiday reduction on Dec 25 vs rejection on invalid date",
    t5_pass,
    f"Dec 25: 24 -> {trips5_valid} (adj {meta5_valid['adjustment_trips']}) | June 15: 24 -> {trips5_invalid} (adj {meta5_invalid['adjustment_trips']}, {meta5_invalid['method_label']})"
)

# --------------------------------------------------------------------------
# Test 6: New Year on December 31 vs invalid date
# --------------------------------------------------------------------------
flt6_valid, trips6_valid, meta6_valid = calculate_defensible_event_prediction(
    baseline_pred=24.0, event_name="New Year", target_date=datetime.date(2014, 12, 31), target_hour=8, station_id=70
)
flt6_invalid, trips6_invalid, meta6_invalid = calculate_defensible_event_prediction(
    baseline_pred=24.0, event_name="New Year", target_date=datetime.date(2015, 1, 1), target_hour=8, station_id=70
)

t6_pass = (
    trips6_valid == 9 and meta6_valid["adjustment_trips"] == -15 and meta6_valid["method_label"] == "Historical Holiday Pattern"
    and trips6_invalid == 24 and meta6_invalid["adjustment_trips"] == 0 and meta6_invalid["method_label"] == "Baseline (Date Mismatch)"
)
record_test(
    6,
    "New Year — holiday reduction on Dec 31 (NYE) vs rejection on Jan 1",
    t6_pass,
    f"Dec 31: 24 -> {trips6_valid} (adj {meta6_valid['adjustment_trips']}) | Jan 1: 24 -> {trips6_invalid} (adj {meta6_invalid['adjustment_trips']}, {meta6_invalid['method_label']})"
)

# --------------------------------------------------------------------------
# Test 7: Concert with Demand Adjustment slider (-50%, 0%, +25%, +100%, and zero baseline)
# --------------------------------------------------------------------------
# Station 70 (baseline 12) tested across -50%, 0%, +25%, +100%
flt7_m50, trips7_m50, meta7_m50 = calculate_defensible_event_prediction(
    baseline_pred=12.0, event_name="Concert", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70, custom_uplift_pct=-0.50
)
flt7_0, trips7_0, meta7_0 = calculate_defensible_event_prediction(
    baseline_pred=12.0, event_name="Concert", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70, custom_uplift_pct=0.0
)
flt7_p25, trips7_p25, meta7_p25 = calculate_defensible_event_prediction(
    baseline_pred=12.0, event_name="Concert", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70, custom_uplift_pct=0.25
)
flt7_p100, trips7_p100, meta7_p100 = calculate_defensible_event_prediction(
    baseline_pred=12.0, event_name="Concert", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70, custom_uplift_pct=1.00
)

# Station 64 (baseline 3) with +25%
flt7_stn64, trips7_stn64, meta7_stn64 = calculate_defensible_event_prediction(
    baseline_pred=3.0, event_name="Concert", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=64, custom_uplift_pct=0.25
)

# Zero baseline (Station 70 at 03:00 AM, baseline 0) with +25% adjustment
flt7_zero, trips7_zero, meta7_zero = calculate_defensible_event_prediction(
    baseline_pred=0.0, event_name="Concert", target_date=datetime.date(2015, 6, 15), target_hour=3, station_id=70, custom_uplift_pct=0.25
)

t7_pass = (
    trips7_m50 == 6 and meta7_m50["adjustment_trips"] == -6
    and trips7_0 == 12 and meta7_0["adjustment_trips"] == 0
    and trips7_p25 == 15 and meta7_p25["adjustment_trips"] == 3
    and trips7_p100 == 24 and meta7_p100["adjustment_trips"] == 12
    and trips7_stn64 == 4 and meta7_stn64["adjustment_trips"] == 1
    and trips7_zero == 0 and meta7_zero["adjustment_trips"] == 0  # Zero baseline never forced positive
    and meta7_p25["method_label"] == "Event Scenario Estimate"
)
record_test(
    7,
    "Concert — Demand Adjustment slider (-50%, 0%, +25%, +100%) and zero baseline preservation",
    t7_pass,
    f"Base 12: -50% -> {trips7_m50} (-6) | 0% -> {trips7_0} (0) | +25% -> {trips7_p25} (+3) | +100% -> {trips7_p100} (+12) | Zero base -> {trips7_zero} (unforced)"
)

# --------------------------------------------------------------------------
# Test 8: Other Event with Demand Adjustment slider (-50%, 0%, +25%, +100%, and zero baseline)
# --------------------------------------------------------------------------
# Base 24 tested across -50%, 0%, +25%, +100%
flt8_m50, trips8_m50, meta8_m50 = calculate_defensible_event_prediction(
    baseline_pred=24.0, event_name="Other Event", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70, custom_uplift_pct=-0.50
)
flt8_0, trips8_0, meta8_0 = calculate_defensible_event_prediction(
    baseline_pred=24.0, event_name="Other Event", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70, custom_uplift_pct=0.0
)
flt8_p25, trips8_p25, meta8_p25 = calculate_defensible_event_prediction(
    baseline_pred=24.0, event_name="Other Event", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70, custom_uplift_pct=0.25
)
flt8_p100, trips8_p100, meta8_p100 = calculate_defensible_event_prediction(
    baseline_pred=24.0, event_name="Other Event", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=70, custom_uplift_pct=1.00
)

# Zero baseline with +100% adjustment
flt8_zero, trips8_zero, meta8_zero = calculate_defensible_event_prediction(
    baseline_pred=0.0, event_name="Other Event", target_date=datetime.date(2015, 6, 15), target_hour=8, station_id=12, custom_uplift_pct=1.00
)

t8_pass = (
    trips8_m50 == 12 and meta8_m50["adjustment_trips"] == -12
    and trips8_0 == 24 and meta8_0["adjustment_trips"] == 0
    and trips8_p25 == 30 and meta8_p25["adjustment_trips"] == 6
    and trips8_p100 == 48 and meta8_p100["adjustment_trips"] == 24
    and trips8_zero == 0 and meta8_zero["adjustment_trips"] == 0  # Zero baseline never forced positive
    and meta8_p25["method_label"] == "Event Scenario Estimate"
)
record_test(
    8,
    "Other Event — Demand Adjustment slider (-50%, 0%, +25%, +100%) and zero baseline preservation",
    t8_pass,
    f"Base 24: -50% -> {trips8_m50} (-12) | 0% -> {trips8_0} (0) | +25% -> {trips8_p25} (+6) | +100% -> {trips8_p100} (+24) | Zero base -> {trips8_zero} (unforced)"
)

# --------------------------------------------------------------------------
# Test 9: End-to-end consistency between prediction, shortage, and redistribution
# --------------------------------------------------------------------------
cap70 = get_station_capacity(70)
avail70 = 20
pred_scenario_trips = trips7_p25  # 15 trips
shortage9 = max(0, pred_scenario_trips - avail70)  # 0 shortage because 20 >= 15
match9_none = find_nearest_safe_donor(
    target_station_id=70, target_dt=ts_peak, dest_demand=pred_scenario_trips, dest_available=avail70, dest_name="SF Caltrain"
)

# Scenario with shortage
avail_low = 10
shortage9_short = max(0, pred_scenario_trips - avail_low)  # 5 bikes needed
match9_donor = find_nearest_safe_donor(
    target_station_id=70, target_dt=ts_peak, dest_demand=pred_scenario_trips, dest_available=avail_low, dest_name="SF Caltrain", safety_buffer=3
)

t9_pass = (
    shortage9 == 0 and match9_none["status"] == "enough_bikes" and match9_none["recommended_transfer"] == 0
    and shortage9_short == 5 and match9_donor["status"] == "safe_donor_found" and match9_donor["recommended_transfer"] == 5
)
record_test(
    9,
    "Redistribution consistency — shortage and donor allocation strictly synchronized with scenario demand",
    t9_pass,
    f"Demand={pred_scenario_trips}: Avail 20 -> shortage={shortage9}, transfer={match9_none['recommended_transfer']} | Avail 10 -> shortage={shortage9_short}, transfer={match9_donor['recommended_transfer']} from {match9_donor['donor_name']}"
)

# --------------------------------------------------------------------------
# Test 10: Event dropdown options validation
# --------------------------------------------------------------------------
expected_dropdown = ["No Event", "Christmas", "New Year", "Concert", "Other Event"]
t10_pass = (EVENT_OPTIONS == expected_dropdown)
record_test(
    10,
    "Event Dropdown specification — contains exactly the 5 required options",
    t10_pass,
    f"Options: {EVENT_OPTIONS}"
)

print("=" * 85)
print("OVERALL RESULT: ALL 10 AUDIT & STATION-SPECIFIC TESTS PASSED SUCCESSFULLY")
print("=" * 85)
