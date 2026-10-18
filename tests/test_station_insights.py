"""
Validation script and test suite for Station Insights calculations across all stations.
Can be run directly via: python test_station_insights.py
"""
import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import get_station_hourly_profile, _fmt_stat, PARQUET_FEATURES_PATH

def test_fmt_stat():
    print("Testing _fmt_stat...")
    assert _fmt_stat(0.0) == "0.0"
    assert _fmt_stat(0.000) == "0.0"
    assert _fmt_stat(15.234) == "15.2"
    assert _fmt_stat(2.789) == "2.79"
    assert _fmt_stat(0.0127) == "0.013"
    assert _fmt_stat(0.04508) == "0.045"
    assert _fmt_stat(0.0004) == "0.000"
    print("  -> _fmt_stat PASSED")

def test_station_21_exact_metrics():
    print("Testing Station 21 exact metrics...")
    st_data = get_station_hourly_profile(21)
    assert st_data is not None, "Station 21 data not found"
    
    # 1. Total historical departures
    assert st_data["total_trips"] == 224, f"Expected 224 trips, got {st_data['total_trips']}"
    
    # 2. Denominator (monitored hours)
    assert st_data["total_hours"] == 17583, f"Expected 17583 hours, got {st_data['total_hours']}"
    
    # 3. Average hourly departures across complete period
    expected_overall = 224.0 / 17583.0
    assert abs(st_data["overall_avg"] - expected_overall) < 1e-5, f"Expected {expected_overall}, got {st_data['overall_avg']}"
    assert _fmt_stat(st_data["overall_avg"]) == "0.013", f"Expected '0.013', got {_fmt_stat(st_data['overall_avg'])}"
    
    # 4. Peak departure hour
    assert st_data["busiest_hour"] == 7, f"Expected peak hour 7 (07:00 AM), got {st_data['busiest_hour']}"
    assert abs(st_data["busiest_hour_demand"] - 0.045081) < 1e-4, f"Expected ~0.04508, got {st_data['busiest_hour_demand']}"
    assert _fmt_stat(st_data["busiest_hour_demand"]) == "0.045", f"Expected '0.045', got {_fmt_stat(st_data['busiest_hour_demand'])}"
    
    # 5. Weekday vs Weekend
    assert abs(st_data["weekday_avg"] - 0.01180) < 1e-4, f"Expected ~0.0118, got {st_data['weekday_avg']}"
    assert abs(st_data["weekend_avg"] - 0.01508) < 1e-4, f"Expected ~0.0151, got {st_data['weekend_avg']}"
    assert _fmt_stat(st_data["weekday_avg"]) == "0.012", f"Expected '0.012', got {_fmt_stat(st_data['weekday_avg'])}"
    assert _fmt_stat(st_data["weekend_avg"]) == "0.015", f"Expected '0.015', got {_fmt_stat(st_data['weekend_avg'])}"
    
    # 6. Profile validation
    assert len(st_data["hourly_df"]) == 24, f"Profile must have 24 hours, got {len(st_data['hourly_df'])}"
    assert list(st_data["hourly_df"]["hour"]) == list(range(24))
    peak_in_profile = st_data["hourly_df"].loc[st_data["hourly_df"]["demand"].idxmax(), "hour"]
    assert peak_in_profile == 7, f"Peak in profile is {peak_in_profile}, expected 7"
    
    print("  -> Station 21 PASSED (All 6 criteria validated)")

def test_all_70_stations():
    print("Testing all stations across the entire dataset...")
    df_stns = pd.read_parquet(PARQUET_FEATURES_PATH, columns=["start_station_id"])
    all_stations = sorted(df_stns["start_station_id"].unique())
    assert len(all_stations) == 70, f"Expected 70 stations, got {len(all_stations)}"
    
    zero_demand_stations = []
    low_demand_stations = []
    high_demand_stations = []
    
    for st_id in all_stations:
        st_data = get_station_hourly_profile(int(st_id))
        assert st_data is not None, f"Failed for station {st_id}"
        
        # Denominator check
        assert st_data["total_hours"] == 17583, f"Unexpected hour count for station {st_id}: {st_data['total_hours']}"
        
        # Departures validity
        assert st_data["total_trips"] >= 0, f"Negative total trips for station {st_id}"
        assert not np.isnan(st_data["overall_avg"]), f"NaN overall_avg for station {st_id}"
        assert not np.isnan(st_data["weekday_avg"]), f"NaN weekday_avg for station {st_id}"
        assert not np.isnan(st_data["weekend_avg"]), f"NaN weekend_avg for station {st_id}"
        
        if st_data["total_trips"] == 0:
            zero_demand_stations.append(st_id)
            assert st_data["overall_avg"] == 0.0
            assert _fmt_stat(st_data["overall_avg"]) == "0.0"
        else:
            assert st_data["overall_avg"] > 0
            formatted_avg = _fmt_stat(st_data["overall_avg"])
            assert formatted_avg != "0.0", (
                f"Station {st_id} has {st_data['total_trips']} trips but average formatted as 0.0"
            )
            assert st_data["busiest_hour_demand"] > 0
            assert _fmt_stat(st_data["busiest_hour_demand"]) != "0.0"
            
            if st_data["overall_avg"] < 0.05:
                low_demand_stations.append((st_id, st_data["total_trips"], formatted_avg))
            else:
                high_demand_stations.append((st_id, st_data["total_trips"], formatted_avg))
                
        # Peak hour valid range 0..23
        assert 0 <= st_data["busiest_hour"] <= 23, f"Invalid peak hour for station {st_id}: {st_data['busiest_hour']}"
        
        # Profile shape & values
        assert len(st_data["hourly_df"]) == 24, f"Profile for station {st_id} does not have 24 hours"
        assert not st_data["hourly_df"]["demand"].isna().any(), f"Profile for station {st_id} contains NaN"
        
        # Peak hour in profile matches busiest_hour and busiest_hour_demand
        max_idx = st_data["hourly_df"]["demand"].idxmax()
        peak_row = st_data["hourly_df"].loc[max_idx]
        assert peak_row["hour"] == st_data["busiest_hour"], (
            f"Station {st_id}: profile peak hour {peak_row['hour']} != card peak hour {st_data['busiest_hour']}"
        )
        assert abs(peak_row["demand"] - st_data["busiest_hour_demand"]) < 1e-5, (
            f"Station {st_id}: profile peak val {peak_row['demand']} != card peak val {st_data['busiest_hour_demand']}"
        )
        
    print(f"  -> All 70 stations PASSED validation!")
    print(f"     Zero-demand stations count: {len(zero_demand_stations)}")
    print(f"     Low-demand stations (<0.05/hr) rescued from 0.0: {len(low_demand_stations)}")
    for s in low_demand_stations:
        print(f"       - Station {s[0]}: {s[1]} trips, formatted avg: {s[2]}/hr")
    print(f"     Standard/High demand stations: {len(high_demand_stations)}")

if __name__ == "__main__":
    test_fmt_stat()
    test_station_21_exact_metrics()
    test_all_70_stations()
    print("\nALL STATION INSIGHTS VALIDATION TESTS COMPLETED SUCCESSFULLY!")
