"""
Unit test verifying conventional nearest-integer, half-up rounding for Rounded Demand Estimate:
- Below 0.5:
  - 0.08 -> 0 trips
  - 0.40 -> 0 trips
  - 0.49 -> 0 trips
- At and above 0.5:
  - 0.50 -> 1 trip
  - 0.75 -> 1 trip
  - 1.49 -> 1 trip
  - 1.50 -> 2 trips
  - 1.60 -> 2 trips
  - 2.50 -> 3 trips
  - 4.03 -> 4 trips
  - 4.50 -> 5 trips
- Genuine zero:
  - 0.00 -> 0 trips
- Negative model outputs (bounded by non-negative constraint):
  - -0.01 -> 0 trips
  - -0.50 -> 0 trips
  - -2.50 -> 0 trips
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import (
    load_production_model,
    load_feature_list,
    load_feature_dataset,
    prepare_prediction_features,
    make_prediction,
    round_half_up,
)

def test_round_half_up_unit():
    print("Testing round_half_up conversion rule...")

    # Genuine zero
    assert round_half_up(0.0) == 0, "0.0 must map to 0"
    assert round_half_up(0.000) == 0, "0.000 must map to 0"
    print("  -> Genuine zero: 0.00 -> 0 [PASS]")

    # User specified examples: 0.08 -> 0, 0.40 -> 0, 0.50 -> 1, 0.75 -> 1, 1.60 -> 2, 4.03 -> 4
    assert round_half_up(0.08) == 0, "0.08 must map to 0"
    assert round_half_up(0.40) == 0, "0.40 must map to 0"
    assert round_half_up(0.50) == 1, "0.50 must map to 1"
    assert round_half_up(0.75) == 1, "0.75 must map to 1"
    assert round_half_up(1.60) == 2, "1.60 must map to 2"
    assert round_half_up(4.03) == 4, "4.03 must map to 4"
    print("  -> User examples (0.08->0, 0.40->0, 0.50->1, 0.75->1, 1.60->2, 4.03->4) [PASS]")

    # Below 0.5 threshold
    assert round_half_up(0.001) == 0, "0.001 must map to 0"
    assert round_half_up(0.49) == 0, "0.49 must map to 0"
    assert round_half_up(0.4999) == 0, "0.4999 must map to 0"
    print("  -> Values below 0.5 threshold: 0.001->0, 0.49->0, 0.4999->0 [PASS]")

    # At 0.5 threshold
    assert round_half_up(0.5) == 1, "0.5 must map to 1"
    assert round_half_up(1.5) == 2, "1.5 must map to 2"
    assert round_half_up(2.5) == 3, "2.5 must map to 3"
    assert round_half_up(3.5) == 4, "3.5 must map to 4"
    print("  -> Values at 0.5 threshold (half-up): 0.5->1, 1.5->2, 2.5->3, 3.5->4 [PASS]")

    # Above 0.5 threshold
    assert round_half_up(0.51) == 1, "0.51 must map to 1"
    assert round_half_up(1.49) == 1, "1.49 must map to 1"
    assert round_half_up(1.51) == 2, "1.51 must map to 2"
    assert round_half_up(4.50) == 5, "4.50 must map to 5"
    assert round_half_up(23.73) == 24, "23.73 must map to 24"
    print("  -> Values above threshold: 0.51->1, 1.49->1, 1.51->2, 4.50->5, 23.73->24 [PASS]")

    # Negative values (non-negative demand constraint)
    assert round_half_up(-0.01) == 0, "-0.01 must map to 0"
    assert round_half_up(-0.50) == 0, "-0.50 must map to 0"
    assert round_half_up(-1.0) == 0, "-1.0 must map to 0"
    assert round_half_up(-10.0) == 0, "-10.0 must map to 0"
    print("  -> Negative values under non-negative constraint: -0.01->0, -0.5->0, -1.0->0, -10.0->0 [PASS]")

def test_live_stations_round_half_up():
    print("\nTesting live model predictions across stations...")
    model = load_production_model()
    features = load_feature_list()
    df_feat = load_feature_dataset()

    # Station 3 (Low demand, rate ~0.088 trips/hr -> 0 trips under half-up)
    ts = pd.Timestamp("2015-06-15 08:00:00")
    X3, _, _ = prepare_prediction_features(3, ts, df_feat, features, "No Event")
    pred3 = make_prediction(model, X3)
    int3 = round_half_up(pred3)
    print(f"  -> Station 3 at 08:00 AM: {pred3:.4f} trips/hr -> {int3} trips (round_half_up)")
    assert 0.0 < pred3 < 0.5
    assert int3 == 0

    # Station 70 at 12:00 PM (Rate ~1.05 trips/hr -> 1 trip under half-up)
    ts_noon = pd.Timestamp("2015-06-15 12:00:00")
    X70_noon, _, _ = prepare_prediction_features(70, ts_noon, df_feat, features, "No Event")
    pred70_noon = make_prediction(model, X70_noon)
    int70_noon = round_half_up(pred70_noon)
    print(f"  -> Station 70 at 12:00 PM: {pred70_noon:.4f} trips/hr -> {int70_noon} trip (round_half_up)")
    assert 1.0 < pred70_noon < 1.5
    assert int70_noon == 1

    # Station 70 at 08:00 AM (Rate ~23.73 trips/hr -> 24 trips under half-up)
    X70_peak, _, _ = prepare_prediction_features(70, ts, df_feat, features, "No Event")
    pred70_peak = make_prediction(model, X70_peak)
    int70_peak = round_half_up(pred70_peak)
    print(f"  -> Station 70 at 08:00 AM: {pred70_peak:.4f} trips/hr -> {int70_peak} trips (round_half_up)")
    assert pred70_peak > 20.0
    assert int70_peak == 24

    # Genuine zero baseline check
    zero_val = 0.0
    int_zero = round_half_up(zero_val)
    print(f"  -> Baseline zero: {zero_val:.2f} trips/hr -> {int_zero} trips")
    assert int_zero == 0

    print("\nALL ROUNDED DEMAND ESTIMATE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_round_half_up_unit()
    test_live_stations_round_half_up()
