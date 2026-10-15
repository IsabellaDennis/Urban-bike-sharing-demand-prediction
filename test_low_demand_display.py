"""
Automated validation of continuous predicted demand rate, separate integer trip estimate,
and accurate demand badges across different demand levels.
"""
import pandas as pd
import numpy as np
from dashboard.app import (
    load_production_model,
    load_feature_list,
    load_feature_dataset,
    prepare_prediction_features,
    make_prediction,
)

def evaluate_display_logic(pred_val: float):
    # Same display formatting logic implemented in app.py
    if pred_val == 0.0:
        formatted_rate = "0.00"
    elif pred_val < 0.01:
        formatted_rate = f"{pred_val:.3f}"
    else:
        formatted_rate = f"{pred_val:.2f}"

    int_estimate = max(0, int(np.floor(float(pred_val) + 0.5)))
    int_unit = "trip" if int_estimate == 1 else "trips"

    if pred_val >= 15.0:
        tag_label = "High Demand"
    elif pred_val >= 5.0:
        tag_label = "Moderate Demand"
    elif pred_val > 0.0:
        tag_label = "Low Demand"
    else:
        tag_label = "Zero Demand"

    return {
        "formatted_rate": formatted_rate,
        "int_estimate": int_estimate,
        "int_unit": int_unit,
        "tag_label": tag_label,
    }

def test_low_demand_display():
    print("Testing Low-Demand Prediction Display Logic...")
    
    # Test case 1: Small positive prediction (e.g. Station 3)
    res_small = evaluate_display_logic(0.07)
    assert res_small["formatted_rate"] == "0.07"
    assert res_small["int_estimate"] == 0
    assert res_small["tag_label"] == "Low Demand", f"Small positive demand must have Low Demand badge, got {res_small['tag_label']}"
    print(f"  -> Small positive (0.07): Rate={res_small['formatted_rate']} trips/hr | Int={res_small['int_estimate']} trips | Badge={res_small['tag_label']} [PASSED]")

    # Test case 2: Genuine zero prediction (0.00)
    res_zero = evaluate_display_logic(0.0)
    assert res_zero["formatted_rate"] == "0.00"
    assert res_zero["int_estimate"] == 0
    assert res_zero["tag_label"] == "Zero Demand", f"Zero demand must have Zero Demand badge, got {res_zero['tag_label']}"
    print(f"  -> Genuine zero (0.00): Rate={res_zero['formatted_rate']} trips/hr | Int={res_zero['int_estimate']} trips | Badge={res_zero['tag_label']} [PASSED]")

    # Test case 3: High demand prediction (23.73)
    res_high = evaluate_display_logic(23.73)
    assert res_high["formatted_rate"] == "23.73"
    assert res_high["int_estimate"] == 24
    assert res_high["tag_label"] == "High Demand"
    print(f"  -> High demand (23.73): Rate={res_high['formatted_rate']} trips/hr | Int={res_high['int_estimate']} trips | Badge={res_high['tag_label']} [PASSED]")

def test_live_model_predictions():
    print("\nTesting Live Model Inference for Target Stations...")
    model = load_production_model()
    features = load_feature_list()
    df_feat = load_feature_dataset()

    # Station 3 at 08:00 AM
    ts = pd.Timestamp("2015-06-15 08:00:00")
    X3, _, _ = prepare_prediction_features(3, ts, df_feat, features, "No Event")
    pred3 = make_prediction(model, X3)
    disp3 = evaluate_display_logic(pred3)
    
    print(f"  -> Station 3 (San Jose Civic Center): Model Rate={disp3['formatted_rate']} trips/hr, Int={disp3['int_estimate']} trips, Badge={disp3['tag_label']}")
    assert pred3 > 0.0, "Station 3 should have positive model rate"
    assert disp3["tag_label"] == "Low Demand", "Station 3 must display Low Demand badge, not Zero Demand"
    assert disp3["int_estimate"] == 0, "Station 3 rounds to 0 integer trips"

    # Station 70 at 08:00 AM
    X70, _, _ = prepare_prediction_features(70, ts, df_feat, features, "No Event")
    pred70 = make_prediction(model, X70)
    disp70 = evaluate_display_logic(pred70)
    print(f"  -> Station 70 (SF Caltrain): Model Rate={disp70['formatted_rate']} trips/hr, Int={disp70['int_estimate']} trips, Badge={disp70['tag_label']}")
    assert pred70 >= 15.0
    assert disp70["tag_label"] == "High Demand"
    assert disp70["int_estimate"] == 24

    print("\nALL PREDICTION DISPLAY TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_low_demand_display()
    test_live_model_predictions()
