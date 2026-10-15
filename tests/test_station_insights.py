"""
Tests for Station Insights calculations and data validation across all stations.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from dashboard.app import load_feature_data, get_station_hourly_profile, _fmt_stat

def test_fmt_stat():
    # Test formatting logic
    assert _fmt_stat(0.0) == "0.0"
    assert _fmt_stat(0.000) == "0.0"
    assert _fmt_stat(15.234) == "15.2"
    assert _fmt_stat(2.789) == "2.79"
    assert _fmt_stat(0.0127) == "0.013"
    assert _fmt_stat(0.04508) == "0.045"
    assert _fmt_stat(0.0004) == "0.000"

def test_station_21_exact_metrics():
    df_feat = load_feature_data()
    assert df_feat is not None and not df_feat.empty
    
    st_data = get_station_hourly_profile(21, df_feat)
    assert st_data is not None
    
    # Total historical departures
    assert st_data["total_trips"] == 224
    # Denominator
    assert st_data["total_hours"] == 17583
    
    # Averages
    expected_overall = 224.0 / 17583.0
    assert pytest.approx(st_data["overall_avg"], rel=1e-4) == expected_overall
    assert _fmt_stat(st_data["overall_avg"]) == "0.013"
    
    # Peak hour
    assert st_data["peak_hour"] == 7
    assert st_data["peak_hour_fmt"] == "07:00 AM"
    assert pytest.approx(st_data["peak_val"], rel=1e-4) == 0.045081
    assert _fmt_stat(st_data["peak_val"]) == "0.045"
    
    # Weekday vs Weekend
    assert pytest.approx(st_data["weekday_avg"], rel=1e-4) == 0.01180
    assert pytest.approx(st_data["weekend_avg"], rel=1e-4) == 0.01508
    assert _fmt_stat(st_data["weekday_avg"]) == "0.012"
    assert _fmt_stat(st_data["weekend_avg"]) == "0.015"
    
    # 24-hour profile has all 24 hours
    assert len(st_data["profile"]) == 24
    assert list(st_data["profile"]["hour"]) == list(range(24))
    
    # Profile peak matches peak_hour
    max_hour = st_data["profile"].loc[st_data["profile"]["demand_dep"].idxmax(), "hour"]
    assert max_hour == 7

def test_all_70_stations():
    df_feat = load_feature_data()
    all_stations = sorted(df_feat["station_id"].unique())
    assert len(all_stations) == 70
    
    for st_id in all_stations:
        st_data = get_station_hourly_profile(st_id, df_feat)
        assert st_data is not None, f"Failed for station {st_id}"
        
        # Denominator check
        assert st_data["total_hours"] == 17583, f"Unexpected hour count for station {st_id}"
        
        # Departures validity
        assert st_data["total_trips"] >= 0
        assert not np.isnan(st_data["overall_avg"])
        assert not np.isnan(st_data["weekday_avg"])
        assert not np.isnan(st_data["weekend_avg"])
        
        # If total_trips > 0, overall_avg must be > 0
        if st_data["total_trips"] > 0:
            assert st_data["overall_avg"] > 0
            assert _fmt_stat(st_data["overall_avg"]) != "0.0", (
                f"Station {st_id} has {st_data['total_trips']} trips but formatted as 0.0"
            )
            assert st_data["peak_val"] > 0
            assert _fmt_stat(st_data["peak_val"]) != "0.0", (
                f"Station {st_id} peak formatted as 0.0"
            )
        else:
            assert st_data["overall_avg"] == 0.0
            assert _fmt_stat(st_data["overall_avg"]) == "0.0"
            
        # Peak hour valid range
        assert 0 <= st_data["peak_hour"] <= 23
        
        # Profile shape
        assert len(st_data["profile"]) == 24
        assert not st_data["profile"]["demand_dep"].isna().any()
        
        # Profile max matches peak_hour and peak_val
        peak_row = st_data["profile"].loc[st_data["profile"]["demand_dep"].idxmax()]
        assert peak_row["hour"] == st_data["peak_hour"]
        assert pytest.approx(peak_row["demand_dep"], rel=1e-5) == st_data["peak_val"]
