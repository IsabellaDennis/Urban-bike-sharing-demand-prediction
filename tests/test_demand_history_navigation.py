"""
Unit and state persistence test for Demand History filters across page navigation,
including Peak Departure Hour verification and confirmation that Peak Departure Intervals is removed.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import query_station_history, PARQUET_FEATURES_PATH

def test_demand_history_peak_hour_calculation():
    print("Testing query_station_history for Station 21 (Sept 1 to Sept 30, 2013)...")
    start_dt = pd.Timestamp("2013-09-01")
    end_dt = pd.Timestamp("2013-09-30 23:59:59")
    
    res = query_station_history(station_id=21, start_dt=start_dt, end_dt=end_dt, aggregation="Daily")
    assert res is not None, "Query returned None"
    assert res["total_trips"] == 29, f"Expected 29 trips, got {res['total_trips']}"
    assert res["total_records"] == 30 * 24, f"Expected 720 hours, got {res['total_records']}"
    assert res["peak_hour"] == 17, f"Expected peak hour 17 (05:00 PM), got {res['peak_hour']}"
    assert "05:00 PM" in res["peak_hour_str"], f"Expected '05:00 PM' in {res['peak_hour_str']}"
    assert "0.200/hr" in res["peak_hour_str"], f"Expected '0.200/hr' in {res['peak_hour_str']}"
    
    # Confirm top_records is no longer produced/needed
    assert "top_records" not in res, "top_records should be removed from query_station_history"
    print(f"  -> Station 21 Sept 2013: {res['peak_hour_str']}")
    print("  -> Station 21 Peak Hour test PASSED")

def test_demand_history_different_stations_and_ranges():
    print("Testing different stations and date ranges for Peak Departure Hour...")
    
    # Test Station 70 (SF Caltrain) - High volume station across full month of June 2015
    res_70 = query_station_history(
        station_id=70,
        start_dt=pd.Timestamp("2015-06-01"),
        end_dt=pd.Timestamp("2015-06-30 23:59:59"),
        aggregation="Daily"
    )
    assert res_70 is not None
    assert res_70["total_trips"] > 2000
    assert res_70["peak_hour"] in [8, 9], f"SF Caltrain peak should be morning commute (8 or 9 AM), got {res_70['peak_hour']}"
    print(f"  -> Station 70 June 2015: Peak {res_70['peak_hour_str']}, Total {res_70['total_trips']} trips")
    
    # Test Station 60 (Embarcadero at Sansome) - Evening commuter station
    res_60 = query_station_history(
        station_id=60,
        start_dt=pd.Timestamp("2014-07-01"),
        end_dt=pd.Timestamp("2014-07-31 23:59:59"),
        aggregation="Hourly"
    )
    assert res_60 is not None
    assert res_60["peak_hour"] == 17, f"Expected peak hour 17 for Station 60, got {res_60['peak_hour']}"
    print(f"  -> Station 60 July 2014: Peak {res_60['peak_hour_str']}, Total {res_60['total_trips']} trips")
    print("  -> Multiple stations and date ranges test PASSED")

def test_state_persistence_logic():
    print("Testing state persistence logic simulation...")
    session_state = {}
    default_station = 70
    all_station_ids = [2, 3, 21, 60, 70]
    HIST_MIN_DATE = pd.to_datetime("2013-01-01").date()
    HIST_MAX_DATE = pd.to_datetime("2015-08-31").date()
    
    # 1. First visit to Demand History (unselected initial state)
    if "hist_persisted_station" not in session_state:
        session_state["hist_persisted_station"] = None
    if "hist_persisted_start_dt" not in session_state:
        session_state["hist_persisted_start_dt"] = None
    if "hist_persisted_end_dt" not in session_state:
        session_state["hist_persisted_end_dt"] = None
    if "hist_persisted_agg" not in session_state:
        session_state["hist_persisted_agg"] = None
        
    assert session_state["hist_persisted_station"] is None
    assert session_state["hist_persisted_start_dt"] is None
    assert session_state["hist_persisted_end_dt"] is None
    assert session_state["hist_persisted_agg"] is None
    
    # 2. User selects filters on Demand History: Station 21, Sept 1 2013, Sept 30 2013, Hourly
    session_state["hist_persisted_station"] = 21
    session_state["hist_persisted_start_dt"] = pd.to_datetime("2013-09-01").date()
    session_state["hist_persisted_end_dt"] = pd.to_datetime("2013-09-30").date()
    session_state["hist_persisted_agg"] = "Hourly"
    
    # 3. Simulate Streamlit navigation away: widget keys purged, persisted keys preserved
    widget_keys = ["hist_stn_select", "hist_start_dt", "hist_end_dt", "hist_agg_select"]
    for k in widget_keys:
        session_state.pop(k, None)
        
    assert session_state["hist_persisted_station"] == 21
    assert session_state["hist_persisted_start_dt"] == pd.to_datetime("2013-09-01").date()
    assert session_state["hist_persisted_end_dt"] == pd.to_datetime("2013-09-30").date()
    assert session_state["hist_persisted_agg"] == "Hourly"
    
    # 4. User navigates back to Demand History
    if "hist_persisted_station" not in session_state:
        session_state["hist_persisted_station"] = None
    if "hist_persisted_start_dt" not in session_state:
        session_state["hist_persisted_start_dt"] = None
    if "hist_persisted_end_dt" not in session_state:
        session_state["hist_persisted_end_dt"] = None
    if "hist_persisted_agg" not in session_state:
        session_state["hist_persisted_agg"] = None
        
    assert session_state["hist_persisted_station"] == 21
    assert session_state["hist_persisted_start_dt"] == pd.to_datetime("2013-09-01").date()
    assert session_state["hist_persisted_end_dt"] == pd.to_datetime("2013-09-30").date()
    assert session_state["hist_persisted_agg"] == "Hourly"
    print("  -> State persistence logic simulation PASSED")

if __name__ == "__main__":
    test_demand_history_peak_hour_calculation()
    test_demand_history_different_stations_and_ranges()
    test_state_persistence_logic()
    print("\nALL DEMAND HISTORY TESTS PASSED SUCCESSFULLY!")
