import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import load_eda_data, get_station_capacity, query_station_history

def test_top_network_stations_calculation():
    print("Testing Top Network Stations calculation...")
    eda_data = load_eda_data()
    assert "station" in eda_data and not eda_data["station"].empty, "station EDA data missing"
    
    df_stn_raw = eda_data["station"].copy()
    
    # Canonical station names
    station_names = {}
    for _, r in df_stn_raw.sort_values("demand", ascending=False).iterrows():
        sid = int(r["start_station_id"])
        if sid not in station_names:
            station_names[sid] = f"Station {sid} — {r['start_station_name']}"
            
    # Grouped by station ID to eliminate variations/duplicates
    df_grouped = df_stn_raw.groupby("start_station_id", as_index=False).agg({"demand": "sum"})
    df_grouped = df_grouped.sort_values("demand", ascending=False).head(10).reset_index(drop=True)
    df_grouped["Rank"] = range(1, len(df_grouped) + 1)
    df_grouped["Total Departures"] = df_grouped["demand"].apply(lambda v: f"{int(v):,}")
    df_grouped["Station Name"] = df_grouped["start_station_id"].apply(
        lambda sid: station_names.get(int(sid), f"Station {int(sid)}")
    )
    df_grouped["Docks"] = df_grouped["start_station_id"].apply(lambda sid: get_station_capacity(int(sid)))
    
    table = df_grouped[["Rank", "Station Name", "Total Departures", "Docks"]]
    print(table.to_string(index=False))
    
    assert len(table) == 10, f"Expected 10 stations, got {len(table)}"
    assert table.iloc[0]["Rank"] == 1
    assert "Station 70" in table.iloc[0]["Station Name"]
    assert table.iloc[0]["Total Departures"] == "49,092"
    assert table.iloc[1]["Rank"] == 2
    assert "Station 69" in table.iloc[1]["Station Name"]
    assert table.iloc[1]["Total Departures"] == "33,742"
    assert table.iloc[2]["Rank"] == 3
    assert "Station 50" in table.iloc[2]["Station Name"]
    assert table.iloc[2]["Total Departures"] == "32,934"
    assert table.iloc[9]["Rank"] == 10
    assert "Station 67" in table.iloc[9]["Station Name"]
    assert table.iloc[9]["Total Departures"] == "20,272"
    
    # Confirm uniqueness
    assert df_grouped["start_station_id"].nunique() == 10, "Duplicate station IDs found in Top 10"
    print("\n[PASS] Top Network Stations calculation verified successfully!")

if __name__ == "__main__":
    test_top_network_stations_calculation()
