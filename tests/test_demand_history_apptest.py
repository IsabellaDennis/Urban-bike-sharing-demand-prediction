import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def test_demand_history_app():
    print("Testing Demand History via Streamlit AppTest...")
    at = AppTest.from_file(str(ROOT_DIR / "dashboard" / "app.py"), default_timeout=30)
    at.run()
    
    # 1. Switch to Demand History page
    print("Selecting Demand History page...")
    nav_radio = at.sidebar.radio[0]
    nav_radio.set_value("Demand History").run()
    
    # Check filters
    print("Checking initial unselected states...")
    stn_sb = at.get("selectbox")[0]
    agg_sb = at.get("selectbox")[1]
    start_dt = at.get("date_input")[0]
    end_dt = at.get("date_input")[1]
    
    print(f"  Station selectbox value: {stn_sb.value}")
    print(f"  Start date value: {start_dt.value}")
    print(f"  End date value: {end_dt.value}")
    print(f"  Agg selectbox value: {agg_sb.value}")
    
    assert stn_sb.value is None, f"Expected Station to be None, got {stn_sb.value}"
    assert start_dt.value is None, f"Expected Start Date to be None, got {start_dt.value}"
    assert end_dt.value is None, f"Expected End Date to be None, got {end_dt.value}"
    assert agg_sb.value is None, f"Expected Time Resolution to be None, got {agg_sb.value}"
    
    # Check that neutral message is displayed and KPI cards are absent
    md_texts = [m.value for m in at.markdown]
    has_neutral_msg = any("Select a station, date range, and time resolution" in t for t in md_texts)
    has_kpis = any("Total Departures in Selected Period" in t for t in md_texts)
    print(f"  Neutral message displayed: {has_neutral_msg}")
    print(f"  KPI cards displayed: {has_kpis}")
    assert has_neutral_msg, "Expected neutral message to be displayed before filters are complete"
    assert not has_kpis, "Expected KPI cards NOT to be displayed initially"
    
    # 2. Select Station 70 only
    print("Selecting Station 70...")
    stn_sb.select(70).run()
    md_texts = [m.value for m in at.markdown]
    assert any("Select a station, date range, and time resolution" in t for t in md_texts), "Neutral message should stay"
    assert not any("Total Departures in Selected Period" in t for t in md_texts), "KPI cards should NOT display yet"
    
    # 3. Select Start Date 2015-06-01
    print("Selecting Start Date 2015-06-01...")
    start_dt = at.get("date_input")[0]
    start_dt.set_value(pd.to_datetime("2015-06-01").date()).run()
    md_texts = [m.value for m in at.markdown]
    assert any("Select a station, date range, and time resolution" in t for t in md_texts), "Neutral message should stay"
    assert not any("Total Departures in Selected Period" in t for t in md_texts), "KPI cards should NOT display yet"
    
    # 4. Select End Date earlier than Start Date (Invalid range test: 2015-05-01 < 2015-06-01)
    print("Testing invalid date range (End Date 2015-05-01 earlier than Start Date 2015-06-01)...")
    end_dt = at.get("date_input")[1]
    agg_sb = at.get("selectbox")[1]
    end_dt.set_value(pd.to_datetime("2015-05-01").date())
    agg_sb.select("Daily").run()
    
    print(f"  Errors rendered: {[e.value for e in at.error]}")
    assert len(at.error) > 0, "Expected error message when start date > end date"
    assert "earlier than or equal to End date" in at.error[0].value
    md_texts = [m.value for m in at.markdown]
    assert not any("Total Departures in Selected Period" in t for t in md_texts), "KPI cards should NOT display with invalid date range"
    
    # 5. Correct End Date to 2015-06-30
    print("Setting valid End Date 2015-06-30...")
    end_dt = at.get("date_input")[1]
    end_dt.set_value(pd.to_datetime("2015-06-30").date()).run()
    
    assert len(at.error) == 0, f"Unexpected errors: {[e.value for e in at.error]}"
    md_texts = [m.value for m in at.markdown]
    has_kpis = any("Total Departures in Selected Period" in t for t in md_texts)
    print(f"  KPI cards displayed after all 4 valid selections: {has_kpis}")
    assert has_kpis, "Expected KPI cards to be displayed when all filters are valid"
    assert not any("Select a station, date range, and time resolution" in t for t in md_texts), "Neutral message should disappear"
    
    # 6. Test Persistence: Navigate to Dashboard and back
    print("Navigating to Dashboard...")
    nav_radio = at.sidebar.radio[0]
    nav_radio.set_value("Dashboard").run()
    
    print("Navigating back to Demand History...")
    nav_radio = at.sidebar.radio[0]
    nav_radio.set_value("Demand History").run()
    
    # Check that selections persisted
    stn_sb = at.get("selectbox")[0]
    agg_sb = at.get("selectbox")[1]
    start_dt = at.get("date_input")[0]
    end_dt = at.get("date_input")[1]
    
    print(f"  Persisted Station: {stn_sb.value}")
    print(f"  Persisted Start Date: {start_dt.value}")
    print(f"  Persisted End Date: {end_dt.value}")
    print(f"  Persisted Aggregation: {agg_sb.value}")
    
    assert stn_sb.value == 70, f"Expected persisted station 70, got {stn_sb.value}"
    assert start_dt.value == pd.to_datetime("2015-06-01").date(), f"Expected persisted start date, got {start_dt.value}"
    assert end_dt.value == pd.to_datetime("2015-06-30").date(), f"Expected persisted end date, got {end_dt.value}"
    assert agg_sb.value == "Daily", f"Expected persisted aggregation Daily, got {agg_sb.value}"
    
    md_texts = [m.value for m in at.markdown]
    has_kpis = any("Total Departures in Selected Period" in t for t in md_texts)
    print(f"  KPI cards still displayed after returning: {has_kpis}")
    assert has_kpis, "Expected KPI cards to remain displayed after returning to Demand History"
    
    print("\nSUCCESS: All Demand History AppTest checks passed!")

if __name__ == "__main__":
    test_demand_history_app()
