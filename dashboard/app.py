# ================================================================
# SMARTBIKE AI - URBAN BIKE-SHARING DEMAND PREDICTION
# Modern Urban Mobility Operations & Analytics Platform
# ================================================================

import base64
import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path for redistribution imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from redistribution.redistribution import (
    calculate_station_shortage_surplus,
    compute_network_rebalancing,
    get_station_capacity,
    estimate_available_bikes,
    find_verified_donor_for_station,
    find_nearest_safe_donor,
    STATION_CAPACITIES,
)

warnings.filterwarnings("ignore")

# Streamlit Page Configuration
st.set_page_config(
    page_title="SmartBike AI — Bike Demand Prediction & Mobility Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Helper function to render HTML cleanly without markdown indentation issues
def render_html(html_str: str):
    clean = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


# ================================================================
# PROJECT-RELATIVE PATH CONFIGURATION
# ================================================================

DATA_DIR = BASE_DIR / "data"
FEATURES_PATH = DATA_DIR / "features" / "station_hourly_demand_features.csv"
PARQUET_FEATURES_PATH = DATA_DIR / "features" / "station_hourly_demand_features.parquet"
AUDITED_FEATURE_LIST_PATH = DATA_DIR / "model_results" / "leakage_audit" / "final_feature_list.csv"
EDA_DIR = DATA_DIR / "eda"
MODELS_DIR = BASE_DIR / "models"
HERO_IMG_PATH = Path(__file__).resolve().parent / "assets" / "smartbike_hero.png"

# Production Model (XGBoost Champion)
PRODUCTION_MODEL_PATH = MODELS_DIR / "xgboost_final_model.pkl"
EXPECTED_FEATURE_COUNT = 54

# Dataset Special Events (strictly 5 required options)
EVENT_OPTIONS = [
    "No Event",
    "Christmas",
    "New Year",
    "Concert",
    "Other Event",
]

# Exact encoding map from trained model pipeline (0 = baseline, 1-4 = event IDs)
EVENT_MAPPING = {
    "No Event": 0,
    "Christmas": 1,
    "New Year": 2,
    "Concert": 4,
    "Other Event": 0,
}

# ================================================================
# SPECIAL EVENT DATA & CALENDAR CONFIGURATION (DEFENSIBLE)
# ================================================================
# Fixed annual holiday calendar dates
FIXED_HOLIDAYS = {
    "Christmas": [(12, 25)],
    "New Year": [(12, 31)],
}

DEFAULT_EVENT_SCHEDULES = {
    "Concert": {
        "scenario_uplift": 0.0,
        "label": "Concert (0% default scenario estimate)",
        "desc": "Concert attendee surge or drop (user-controlled scenario)",
    },
    "Other Event": {
        "scenario_uplift": 0.0,
        "label": "Other Event (0% default scenario estimate)",
        "desc": "Special community or cultural gathering (user-controlled scenario)",
    },
    "Christmas": {
        "scenario_uplift": -0.75,
        "label": "Christmas (-75% holiday commute drop)",
        "desc": "Observed holiday commercial/office closure",
    },
    "New Year": {
        "scenario_uplift": -0.62,
        "label": "New Year (-62% holiday commute drop)",
        "desc": "Observed New Year's Eve holiday closure (Dec 31)",
    },
}

CORRIDOR_STATION_IDS = {70, 69, 61, 50, 60, 55, 74, 77, 65, 76, 39, 72}

STATION_SHORT_NAMES = {
    70: "SF Caltrain (Townsend at 4th)",
    69: "San Francisco Caltrain 2",
    60: "Embarcadero at Sansome",
    50: "Harry Bridges Plaza (Ferry Building)",
    61: "2nd at Townsend",
    55: "Temporary Transbay Terminal",
    74: "Steuart at Market",
    77: "Market at Sansome",
    65: "Townsend at 7th",
    67: "Market at 10th",
    76: "Market at 4th",
    64: "2nd at South Park",
    39: "Powell Street BART",
    73: "Grant at Columbus",
    62: "2nd at Folsom",
    56: "Beale at Market",
    54: "Embarcadero at Bryant",
    72: "Civic Center BART",
    51: "Embarcadero at Folsom",
    47: "Post at Kearny",
    48: "Embarcadero at Vallejo",
    45: "Commercial at Montgomery",
}


def format_station_label(sid: int, raw_name: str = None) -> str:
    """Format clean readable station label."""
    if sid in STATION_SHORT_NAMES:
        return f"Station {sid} — {STATION_SHORT_NAMES[sid]}"
    if raw_name:
        clean = str(raw_name).replace(f"Station {sid} — ", "").replace(f"Station {sid} - ", "").strip()
        return f"Station {sid} — {clean}"
    return f"Station {sid}"


def format_bikes(n: int) -> str:
    if n is None:
        return "—"
    val = abs(int(n))
    return "1 bike" if val == 1 else f"{val} bikes"


def format_departures(n: int) -> str:
    if n is None:
        return "—"
    val = abs(int(n))
    return "1 departure" if val == 1 else f"{val} departures"


def format_trips(n: int) -> str:
    if n is None:
        return "—"
    val = abs(int(n))
    return "1 trip" if val == 1 else f"{val} trips"


def format_docks(n: int) -> str:
    if n is None:
        return "—"
    val = abs(int(n))
    return "1 dock" if val == 1 else f"{val} docks"


# ================================================================
# BACKEND DATA & MODEL LOADING
# ================================================================

@st.cache_resource
def load_production_model():
    """Load the trained XGBoost production model."""
    if PRODUCTION_MODEL_PATH.exists():
        with open(PRODUCTION_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_feature_list():
    """Load the authoritative 54-feature schema verified during audit."""
    model = load_production_model()
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if AUDITED_FEATURE_LIST_PATH.exists():
        df_feat = pd.read_csv(AUDITED_FEATURE_LIST_PATH)
        col = [c for c in ["Feature", "feature", "features", "column"] if c in df_feat.columns][0]
        return df_feat[col].dropna().astype(str).tolist()
    return []


@st.cache_data
def load_eda_data():
    """Load pre-computed historical EDA summary tables."""
    eda = {}
    files = {
        "hourly": "hourly_demand.csv",
        "daily": "daily_demand.csv",
        "weekday": "weekday_demand.csv",
        "weekday_hour": "weekday_hour_demand.csv",
        "day_type": "weekday_vs_weekend.csv",
        "monthly": "monthly_demand.csv",
        "station": "station_demand.csv",
    }
    for key, fname in files.items():
        p = EDA_DIR / fname
        if p.exists():
            eda[key] = pd.read_csv(p)
    return eda


@st.cache_data
def load_feature_dataset():
    """Load feature dataset via high-performance Parquet format."""
    if PARQUET_FEATURES_PATH.exists():
        df = pd.read_parquet(PARQUET_FEATURES_PATH)
        if not pd.api.types.is_datetime64_any_dtype(df["hour_timestamp"]):
            df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"], errors="coerce")
        return df
    elif FEATURES_PATH.exists():
        df = pd.read_csv(FEATURES_PATH)
        df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"], errors="coerce")
        return df
    return None


@st.cache_data
def get_station_hourly_profile(station_id: int):
    """Compute complete 24-hour average hourly demand profile and summary statistics for a station."""
    if PARQUET_FEATURES_PATH.exists():
        cols = ["start_station_id", "hour", "demand", "is_weekday"]
        df = pd.read_parquet(PARQUET_FEATURES_PATH, columns=cols)
        subset = df[df["start_station_id"] == station_id]
        if not subset.empty:
            total_trips = int(subset["demand"].sum())
            total_hours = len(subset)
            overall_avg = float(subset["demand"].mean())
            
            # Guarantee all 24 hours (0-23) are explicitly represented
            hourly_avg = subset.groupby("hour")["demand"].mean().reindex(range(24), fill_value=0.0).reset_index()
            
            weekday_sub = subset[subset["is_weekday"] == 1]
            weekend_sub = subset[subset["is_weekday"] == 0]
            
            weekday_avg = float(weekday_sub["demand"].mean()) if not weekday_sub.empty else 0.0
            weekend_avg = float(weekend_sub["demand"].mean()) if not weekend_sub.empty else 0.0
            
            busiest_hour_row = hourly_avg.loc[hourly_avg["demand"].idxmax()]
            return {
                "hourly_df": hourly_avg,
                "total_trips": total_trips,
                "total_hours": total_hours,
                "overall_avg": overall_avg,
                "weekday_avg": weekday_avg,
                "weekend_avg": weekend_avg,
                "busiest_hour": int(busiest_hour_row["hour"]),
                "busiest_hour_demand": float(busiest_hour_row["demand"]),
            }
    return None


HOUR_LABELS = [
    "12:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", "05:00 AM",
    "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
    "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM",
    "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"
]


def _fmt_stat(val: float) -> str:
    """Format demand statistics with adaptive decimal precision to prevent spurious 0.0 values."""
    if val == 0.0:
        return "0.0"
    elif val >= 10.0:
        return f"{val:.1f}"
    elif val >= 1.0:
        return f"{val:.2f}"
    else:
        return f"{val:.3f}"


@st.cache_data
def query_station_history(station_id: int, start_dt: pd.Timestamp, end_dt: pd.Timestamp, aggregation: str = "Daily"):
    """Extract and aggregate historical records for Demand History page."""
    if PARQUET_FEATURES_PATH.exists():
        cols = ["start_station_id", "hour_timestamp", "demand", "is_weekday", "hour"]
        df = pd.read_parquet(PARQUET_FEATURES_PATH, columns=cols)
        sub = df[
            (df["start_station_id"] == station_id) &
            (df["hour_timestamp"] >= start_dt) &
            (df["hour_timestamp"] <= end_dt)
        ].sort_values("hour_timestamp")

        if sub.empty:
            return None

        total_trips = int(sub["demand"].sum())
        total_hours = len(sub)
        weekday_sub = sub[sub["is_weekday"] == 1]
        weekend_sub = sub[sub["is_weekday"] == 0]

        weekday_trips = int(weekday_sub["demand"].sum()) if not weekday_sub.empty else 0
        weekend_trips = int(weekend_sub["demand"].sum()) if not weekend_sub.empty else 0

        if aggregation == "Daily":
            trend_df = sub.set_index("hour_timestamp").resample("D")["demand"].sum().reset_index()
            trend_df.columns = ["Timestamp", "Departures"]
            avg_val = trend_df["Departures"].mean() if not trend_df.empty else 0.0
        else:
            trend_df = sub[["hour_timestamp", "demand"]].copy()
            trend_df.columns = ["Timestamp", "Departures"]
            avg_val = trend_df["Departures"].mean() if not trend_df.empty else 0.0

        # Hourly average departure profile across selected period
        hourly_series = sub.groupby("hour")["demand"].mean().reindex(range(24), fill_value=0.0)
        busiest_hour_int = int(hourly_series.idxmax())
        busiest_hour_demand = float(hourly_series.loc[busiest_hour_int])
        peak_hour_str = f"{HOUR_LABELS[busiest_hour_int]} ({_fmt_stat(busiest_hour_demand)}/hr)"

        return {
            "trend_df": trend_df,
            "total_trips": total_trips,
            "total_records": total_hours,
            "avg_departures": avg_val,
            "peak_hour": busiest_hour_int,
            "peak_hour_demand": busiest_hour_demand,
            "peak_hour_str": peak_hour_str,
            "weekday_trips": weekday_trips,
            "weekend_trips": weekend_trips,
        }
    return None


def get_hero_base64():
    """Load base64-encoded bicycle hero image."""
    if HERO_IMG_PATH.exists():
        with open(HERO_IMG_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def validate_features(features_df: pd.DataFrame, expected_features: list) -> tuple:
    """Validate that features strictly adhere to schema and contain no target leakage."""
    if features_df is None or features_df.empty:
        return False, "Feature dataframe is empty or unavailable."
    if len(features_df.columns) != EXPECTED_FEATURE_COUNT:
        return False, f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, got {len(features_df.columns)}."
    if "demand" in features_df.columns:
        return False, "Target leakage detected: 'demand' column present."
    if features_df.isna().sum().sum() > 0:
        return False, "Missing values detected in features."
    return True, ""


def prepare_prediction_features(
    station_id: int,
    target_ts: pd.Timestamp,
    df_features: pd.DataFrame,
    expected_features: list,
    special_event: str = "No Event"
) -> tuple:
    """Extract historical feature snapshot for genuine model inference."""
    if df_features is None:
        return None, None, "Feature dataset is not accessible."

    record = df_features[
        (df_features["start_station_id"] == int(station_id)) &
        (df_features["hour_timestamp"] == target_ts)
    ]

    if record.empty:
        return None, None, "Timestamp is outside historical training records; using statistical baseline."

    X_sample = record[expected_features].copy()
    actual_val = float(record["demand"].values[0]) if "demand" in record.columns else None

    if "special_event" in expected_features:
        X_sample["special_event"] = EVENT_MAPPING.get(special_event, 0)

    is_valid, err_msg = validate_features(X_sample, expected_features)
    if not is_valid:
        return None, None, f"Feature validation failed: {err_msg}"

    return X_sample, actual_val, "OK"


def make_prediction(model, X_sample: pd.DataFrame) -> float:
    """Generate physically-bounded non-negative demand forecast."""
    pred_raw = model.predict(X_sample)[0]
    return max(float(pred_raw), 0.0)


def check_date_compatibility(event_name: str, target_date) -> tuple:
    """Verify if a selected fixed holiday matches the calendar date."""
    if event_name == "Christmas":
        if (target_date.month, target_date.day) == (12, 25):
            return True, "Valid Christmas holiday date (December 25)."
        return False, f"Date mismatch: Christmas is observed on December 25, but selected date is {target_date.strftime('%Y-%m-%d')}."
    elif event_name == "New Year":
        if (target_date.month, target_date.day) == (12, 31):
            return True, "Valid New Year date (December 31 is the date represented by the historical New Year labels in this dataset)."
        return False, f"Date mismatch: In this historical dataset, New Year is tagged strictly on December 31 (New Year's Eve), but selected date is {target_date.strftime('%Y-%m-%d')}."
    return True, "Event applies without fixed annual calendar date restrictions."
def round_half_up(val: float) -> int:
    """Conventional nearest-integer, half-up rounding with non-negative lower bound."""
    return int(np.floor(max(0.0, float(val)) + 0.5))


def calculate_defensible_event_prediction(
    baseline_pred: float,
    event_name: str,
    target_date,
    target_hour: int,
    station_id: int,
    custom_uplift_pct: float = None,
    min_scenario_demand: int = 0,
    event_start_hour: int = None,
    event_end_hour: int = None,
    event_scope: str = "all",
    is_hypothetical_override: bool = False,
) -> tuple:
    """
    Station-specific and defensible event-aware demand prediction.

    Rules:
      1. No Event:
         - Show normal station-specific baseline prediction without any adjustment.
         - Genuine zero prediction remains zero.
      2. Christmas:
         - Apply configured holiday reduction (-75%) ONLY on December 25.
         - On any other date, show date-mismatch warning; no holiday adjustment applied.
      3. New Year:
         - Apply configured holiday reduction (-62%) ONLY on December 31.
         - On any other date, show date-mismatch warning; no holiday adjustment applied.
         - Explains that December 31 is the date represented by historical New Year labels.
      4. Concert & Other Event:
         - General Demand Adjustment (%) slider from -50% to +100% (default 0%).
         - Negative values decrease demand, zero leaves demand unchanged, positive values increase demand.
         - Calculated as baseline * (1 + adjustment percentage / 100), rounded half-up, non-negative.
         - Label result 'Event Scenario Estimate' and explain that the percentage is a user-controlled
           planning assumption, not an effect learned by the ML model.
         - Do not force a positive prediction when baseline is zero. Show a short explanation in that case.
    """
    def _round_half_up(val: float) -> int:
        return int(np.floor(float(val) + 0.5))

    clean_base = max(0.0, float(baseline_pred))
    base_trips = _round_half_up(clean_base)

    # 1. No Event
    if event_name == "No Event" or not event_name:
        return clean_base, base_trips, {
            "event_name": "No Event",
            "is_event_applied": False,
            "event_type": "none",
            "baseline_trips": base_trips,
            "adjustment_trips": 0,
            "final_trips": base_trips,
            "baseline_float": clean_base,
            "final_float": clean_base,
            "adjustment_pct": 0,
            "method_label": "Verified ML Model Baseline",
            "badge_color": "#2563EB",
            "explanation": "Standard prediction without event adjustment.",
        }

    # 2. Christmas
    if event_name == "Christmas":
        is_compat, msg = check_date_compatibility("Christmas", target_date)
        if not is_compat:
            return clean_base, base_trips, {
                "event_name": "Christmas",
                "is_event_applied": False,
                "event_type": "date_mismatch",
                "baseline_trips": base_trips,
                "adjustment_trips": 0,
                "final_trips": base_trips,
                "baseline_float": clean_base,
                "final_float": clean_base,
                "adjustment_pct": 0,
                "method_label": "Baseline (Date Mismatch)",
                "badge_color": "#D97706",
                "explanation": "Invalid date for this holiday. No adjustment applied.",
            }
        uplift = -0.75 if custom_uplift_pct is None else custom_uplift_pct
        adj_float = max(0.0, clean_base * (1.0 + uplift))
        final_trips = max(0, _round_half_up(adj_float))
        diff = final_trips - base_trips
        return adj_float, final_trips, {
            "event_name": "Christmas",
            "is_event_applied": True,
            "event_type": "holiday_reduction",
            "baseline_trips": base_trips,
            "adjustment_trips": diff,
            "final_trips": final_trips,
            "baseline_float": clean_base,
            "final_float": adj_float,
            "adjustment_pct": int(round(uplift * 100)),
            "method_label": "Historical Holiday Pattern",
            "badge_color": "#0284C7",
            "explanation": "Christmas holiday adjustment applied.",
        }

    # 3. New Year
    if event_name == "New Year":
        is_compat, msg = check_date_compatibility("New Year", target_date)
        if not is_compat:
            return clean_base, base_trips, {
                "event_name": "New Year",
                "is_event_applied": False,
                "event_type": "date_mismatch",
                "baseline_trips": base_trips,
                "adjustment_trips": 0,
                "final_trips": base_trips,
                "baseline_float": clean_base,
                "final_float": clean_base,
                "adjustment_pct": 0,
                "method_label": "Baseline (Date Mismatch)",
                "badge_color": "#D97706",
                "explanation": "Invalid date for this holiday. No adjustment applied.",
            }
        uplift = -0.62 if custom_uplift_pct is None else custom_uplift_pct
        adj_float = max(0.0, clean_base * (1.0 + uplift))
        final_trips = max(0, _round_half_up(adj_float))
        diff = final_trips - base_trips
        return adj_float, final_trips, {
            "event_name": "New Year",
            "is_event_applied": True,
            "event_type": "holiday_reduction",
            "baseline_trips": base_trips,
            "adjustment_trips": diff,
            "final_trips": final_trips,
            "baseline_float": clean_base,
            "final_float": adj_float,
            "adjustment_pct": int(round(uplift * 100)),
            "method_label": "Historical Holiday Pattern",
            "badge_color": "#0284C7",
            "explanation": "New Year holiday adjustment applied.",
        }

    # 4. Concert & Other Event (general Demand Adjustment slider from -50% to +100%, default 0%)
    if event_name in ["Concert", "Other Event"]:
        uplift = 0.0 if custom_uplift_pct is None else float(custom_uplift_pct)
        adj_percent_int = int(round(uplift * 100))

        if clean_base > 0.0 and base_trips > 0:
            adj_float = max(0.0, clean_base * (1.0 + uplift))
            final_trips = max(0, _round_half_up(adj_float))
            diff = final_trips - base_trips
        else:
            final_trips = 0
            adj_float = 0.0
            diff = 0

        if adj_percent_int == 0:
            expl = "Demand unchanged from baseline."
        elif event_name == "Concert":
            expl = f"Concert scenario: demand adjusted by {adj_percent_int:+d}%."
        else:
            expl = f"Other event scenario: demand adjusted by {adj_percent_int:+d}%."

        return adj_float, final_trips, {
            "event_name": event_name,
            "is_event_applied": True,
            "event_type": "event_scenario",
            "baseline_trips": base_trips,
            "adjustment_trips": diff,
            "final_trips": final_trips,
            "baseline_float": clean_base,
            "final_float": adj_float,
            "adjustment_pct": adj_percent_int,
            "method_label": "Event Scenario Estimate",
            "badge_color": "#4F46E5",
            "explanation": expl,
        }

    # Fallback default
    return clean_base, base_trips, {
        "event_name": event_name,
        "is_event_applied": False,
        "event_type": "none",
        "baseline_trips": base_trips,
        "adjustment_trips": 0,
        "final_trips": base_trips,
        "baseline_float": clean_base,
        "final_float": clean_base,
        "adjustment_pct": 0,
        "method_label": "Verified ML Model Baseline",
        "badge_color": "#2563EB",
        "explanation": "Standard prediction without event adjustment.",
    }


def get_station_baseline(selected_stn: int, eda_data: dict) -> float:
    """Calculate genuine historical baseline hourly departures for a station."""
    if "station" in eda_data and not eda_data["station"].empty:
        df_st = eda_data["station"]
        match = df_st[df_st["start_station_id"] == selected_stn]
        if not match.empty:
            total_trips = float(match["demand"].values[0])
            return round(total_trips / 17583.0, 1)
    return 3.0


# ================================================================
# REAL WEBSITE ENTERPRISE CSS DESIGN SYSTEM (EQUAL SIZE & BEAUTIFUL CARDS)
# ================================================================

def build_custom_css(b64_hero: str = "") -> str:
    hero_bg = (
        f"""
        background-image: linear-gradient(90deg,
            rgba(10, 17, 34, 0.88) 0%,
            rgba(10, 17, 34, 0.72) 30%,
            rgba(10, 17, 34, 0.28) 52%,
            rgba(10, 17, 34, 0.05) 70%,
            rgba(10, 17, 34, 0.0) 84%
        ), url('data:image/png;base64,{b64_hero}') !important;
        """
        if b64_hero
        else "background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 100%) !important;"
    )

    css_code = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

h1, h2, h3, h4, .headline-font {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ================================================================
   MAIN CANVAS & DISTINCT DASHBOARD COLOR SCHEME
   ================================================================ */
[data-testid="stAppViewContainer"] {
    background-color: #EEF2F6 !important;
    background-image: none !important;
    color: #0F172A !important;
    overflow-x: hidden !important;
}

/* Chrome removal & zero top offsets */
header[data-testid="stHeader"],
div[data-testid="stApp"] header,
header {
    display: none !important;
    visibility: hidden !important;
    height: 0px !important;
    min-height: 0px !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stSidebarHeader"] {
    display: none !important;
    height: 0px !important;
}

div[data-testid="stAppViewContainer"] > section[data-testid="stMain"],
div[data-testid="stAppViewContainer"] > section.main {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

div[data-testid="stMainBlockContainer"],
div[data-testid="block-container"],
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 100% !important;
}

/* Hide markdown/style wrappers that take up vertical flex space */
div[data-testid="stElementContainer"]:has(style),
div[data-testid="stElementContainer"]:empty {
    display: none !important;
    height: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Ensure first content in main starts flush at top */
div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Clean Enterprise Dark Slate Sidebar */
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 275px !important;
    width: 268px !important;
    background: #090E1A !important;
    border-right: 1px solid #1E293B !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #94A3B8;
}

/* Sidebar Flex-Column Layout */
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    height: 100% !important;
}

[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    height: 100% !important;
    min-height: calc(100vh - 56px) !important;
    justify-content: flex-start !important;
    box-sizing: border-box !important;
}

[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
    padding: 18px 14px 28px 14px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    box-sizing: border-box !important;
}

[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.sidebar-bottom-anchor) {
    margin-top: auto !important;
    width: 100% !important;
    padding-bottom: 8px !important;
}

/* Sidebar Nav Radio Options - Enlarged and prominently styled */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 8px !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background-color: transparent !important;
    color: #94A3B8 !important;
    padding: 12px 14px !important;
    border-radius: 9px !important;
    font-size: 0.94rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
    border: 1px solid transparent !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border-color: #334155 !important;
}

/* Active Nav: Corporate blue with readable white text */
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #1D4ED8 !important;
    border: 1px solid #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(29, 78, 216, 0.25) !important;
    transform: none !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ================================================================
   CRITICAL: ENFORCE EQUAL CARD HEIGHTS & CRISP WHITE CARDS
   ================================================================ */
/* Streamlit column row layout flex stretch */
div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 0px !important;
    align-self: stretch !important;
    height: 100% !important;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div.stVerticalBlock,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 100% !important;
    height: 100% !important;
    align-self: stretch !important;
}

/* Pure Crisp White SaaS Card Container */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 100% !important;
    height: 100% !important;
    margin-bottom: 0px !important;
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03) !important;
    padding: 24px 26px !important;
    transition: all 0.22s ease !important;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"]:hover,
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #93C5FD !important;
    box-shadow: 0 12px 28px -4px rgba(15, 23, 42, 0.09), 0 4px 10px -2px rgba(37, 99, 235, 0.06) !important;
    transform: translateY(-2px);
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 100% !important;
    height: 100% !important;
    justify-content: space-between !important;
    background: transparent !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: transparent !important;
}

/* Equal-height border container flex cards */
div[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100% !important;
}

/* Hero Panoramic Banner */
.hero-banner-wrap {
    position: relative;
    width: calc(100% + 4rem) !important;
    margin-left: -2rem !important;
    margin-right: -2rem !important;
    margin-top: 0rem !important;
    min-height: 165px;
    height: 165px;
    __HERO_BG__
    background-size: cover !important;
    background-position: right 62% !important;
    background-repeat: no-repeat !important;
    background-color: #0F172A !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding-left: 36px;
    padding-right: 36px;
    padding-top: 14px;
    padding-bottom: 14px;
    margin-bottom: 22px !important;
    border-bottom: 1px solid #1E293B;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.12);
}

.hero-heading {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.95rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 3px 0 !important;
    line-height: 1.15 !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.70) !important;
}

.hero-sub {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    color: #93C5FD !important;
    margin: 0 0 4px 0 !important;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.25) !important;
}

.hero-body {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    color: #E0E7FF !important;
    max-width: 600px !important;
    line-height: 1.35 !important;
    margin: 0 !important;
    opacity: 0.95;
}

/* Section Titles & Subtitles */
.page-section-header {
    margin-bottom: 16px;
}

.page-section-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.40rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.01em;
}

.page-section-sub {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem;
    color: #64748B;
    margin-top: 3px;
}

/* Card Top Title Row */
.card-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}

.card-title-group {
    display: flex;
    align-items: center;
    gap: 14px;
}

.card-icon-box {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
}

.card-title-main {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.20rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.01em;
    line-height: 1.2;
}

.card-title-sub {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 2px;
}

/* Crisp Clean Form Controls */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 4px !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
div[data-testid="stDateInput"] div[data-baseweb="input"] > div {
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
    min-height: 44px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color: #2563EB !important;
}

div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
div[data-baseweb="select"] span {
    color: #0F172A !important;
    font-weight: 600 !important;
}

/* Electric Blue Action Button */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.98rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 13px 24px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease !important;
    margin-top: 6px !important;
    width: 100% !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-2px);
}

/* Secondary preset buttons */
div.stButton > button[kind="secondary"] {
    background: #F8FAFC !important;
    color: #475569 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 6px 10px !important;
    transition: all 0.15s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
    background: #EFF6FF !important;
    border-color: #93C5FD !important;
    color: #1D4ED8 !important;
}

/* Modern SaaS Status Badges */
.badge-green {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    color: #065F46;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
}

.badge-amber {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    color: #92400E;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
}

.badge-red {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FEF2F2;
    border: 1px solid #FECACA;
    color: #991B1B;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
}

.badge-blue {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1E40AF;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
}

.badge-purple {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F5F3FF;
    border: 1px solid #DDD6FE;
    color: #6D28D9;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* Metric Spec Rows */
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.86rem;
}

.metric-row:last-child {
    border-bottom: none;
}

.metric-label {
    color: #64748B;
    font-weight: 500;
}

.metric-val {
    color: #0F172A;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}

/* UNIFORM EQUAL HEIGHT KPI CARDS (STRICT 140px) */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 140px;
    max-height: 140px;
    height: 140px;
    box-sizing: border-box;
    position: relative;
    overflow: hidden;
    transition: all 0.22s ease;
}

.kpi-card:hover {
    box-shadow: 0 10px 24px -2px rgba(15, 23, 42, 0.09);
    border-color: #93C5FD;
    transform: translateY(-2px);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
}

.kpi-card-blue::before { background: linear-gradient(90deg, #2563EB, #60A5FA); }
.kpi-card-green::before { background: linear-gradient(90deg, #059669, #34D399); }
.kpi-card-amber::before { background: linear-gradient(90deg, #D97706, #FBBF24); }
.kpi-card-purple::before { background: linear-gradient(90deg, #7C3AED, #A78BFA); }

.kpi-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.70rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.kpi-icon-pill {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    flex-shrink: 0;
}

.kpi-val-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin: 2px 0;
    white-space: nowrap;
}

.kpi-val-num {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.15rem;
    font-weight: 800;
    color: #0F172A;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
}

.kpi-val-unit {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.20rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
}

.kpi-number {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.05rem;
    font-weight: 800;
    color: #0F172A;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
    margin: 2px 0;
}

.kpi-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #64748B;
    line-height: 1.25;
}

/* Mini 3-Stat Blocks */
.stat-tri-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 14px 0;
}

.stat-tri-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 14px;
    text-align: center;
    transition: all 0.2s ease;
}

.stat-tri-box:hover {
    border-color: #CBD5E1;
    background: #F1F5F9;
}

.stat-tri-num {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 3px 0;
}

.stat-tri-lbl {
    font-size: 0.72rem;
    color: #64748B;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Availability 3-Box Grid */
.avail-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 14px 0 10px 0;
}

@media (max-width: 768px) {
    .avail-grid {
        grid-template-columns: 1fr;
    }
}

.avail-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}

.avail-box-lbl {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.avail-box-num {
    display: flex;
    align-items: baseline;
    justify-content: center;
    gap: 6px;
    white-space: nowrap;
}

.avail-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.1;
}

.avail-unit {
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    font-weight: 500;
    color: #64748B !important;
    line-height: 1.1;
    margin-left: 2px;
}

/* Horizontal Redistribution Flow */
.redistrib-flow-container {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 14px;
    margin: 14px 0 10px 0;
}

@media (max-width: 768px) {
    .redistrib-flow-container {
        flex-direction: column;
    }
}

.redistrib-box-from {
    flex: 1;
    background: #ECFDF5;
    border: 1.5px solid #A7F3D0;
    border-radius: 12px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.redistrib-box-move {
    width: 145px;
    background: #EFF6FF;
    border: 1.5px solid #BFDBFE;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.redistrib-box-to {
    flex: 1;
    background: #FEF2F2;
    border: 1.5px solid #FECACA;
    border-radius: 12px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* EQUAL SIZE CONTENT WRAPPERS */
.equal-card-inner-top {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 470px;
    height: 100%;
}

.equal-card-inner-bottom {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 450px;
    height: 100%;
}

/* ================================================================
   ABOUT PROJECT: 2x2 EQUAL-HEIGHT GRID SYSTEM
   ================================================================ */
.about-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    grid-auto-rows: 1fr;
    gap: 18px;
    width: 100%;
}

@media (max-width: 820px) {
    .about-grid {
        grid-template-columns: 1fr;
        grid-auto-rows: auto;
    }
}

.about-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    min-height: 290px;
    box-sizing: border-box;
    transition: all 0.22s ease;
    position: relative;
    overflow: hidden;
}

.about-card:hover {
    box-shadow: 0 10px 24px -2px rgba(15, 23, 42, 0.08);
    border-color: #93C5FD;
    transform: translateY(-2px);
}

.about-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3.5px;
}

.about-card-blue::before { background: linear-gradient(90deg, #2563EB, #60A5FA); }
.about-card-purple::before { background: linear-gradient(90deg, #7C3AED, #A78BFA); }
.about-card-emerald::before { background: linear-gradient(90deg, #059669, #34D399); }
.about-card-cyan::before { background: linear-gradient(90deg, #0284C7, #38BDF8); }

.about-header-row {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-bottom: 14px;
}

.about-icon-box {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
}

.about-card-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.12rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.01em;
    line-height: 1.25;
}

.about-card-body {
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.6;
    flex-grow: 1;
}

.about-list {
    margin: 0;
    padding-left: 18px;
    list-style-type: disc;
}

.about-list li {
    font-size: 0.87rem;
    color: #475569;
    line-height: 1.6;
    margin-bottom: 7px;
}

.about-list li:last-child {
    margin-bottom: 0;
}

.about-note-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 9px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #64748B;
    line-height: 1.45;
    margin-top: 14px;
}

.about-footer-tag {
    font-size: 0.75rem;
    color: #64748B;
    border-top: 1px solid #F1F5F9;
    padding-top: 10px;
    margin-top: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.about-chip-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 12px;
}

.about-chip {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 9px;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
}

.about-chip-lbl {
    font-size: 0.67rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 2px;
}

.about-chip-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.90rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
}

.tech-tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
}

.tech-tag {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1E40AF;
    font-size: 0.73rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
}
</style>
"""
    return css_code.replace("__HERO_BG__", hero_bg)


# ================================================================
# MAIN APPLICATION CONTROLLER
# ================================================================

def main():
    b64_hero = get_hero_base64()

    # Core models and datasets
    production_model = load_production_model()
    audited_features = load_feature_list()
    eda_data = load_eda_data()

    # Station labels mapping with canonical names (highest-demand variant per station ID)
    station_names = {}
    if "station" in eda_data and not eda_data["station"].empty:
        for _, r in eda_data["station"].sort_values("demand", ascending=False).iterrows():
            sid = int(r["start_station_id"])
            if sid not in station_names:
                station_names[sid] = f"Station {sid} — {r['start_station_name']}"

    all_station_ids = sorted(list(station_names.keys())) if station_names else list(range(2, 85))
    default_station = 70 if 70 in all_station_ids else all_station_ids[0]

    # Hour labels
    hour_labels = [
        "12:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", "05:00 AM",
        "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
        "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM",
        "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"
    ]
    # Hour labels and options (clean 12-hour format without peak annotations)
    hour_options = list(hour_labels)
    hour_to_int = {lbl: idx for idx, lbl in enumerate(hour_options)}

    # ------------------------------------------------------------
    # SIDEBAR: CLEAN ENTERPRISE BRANDING & NAVIGATION
    # ------------------------------------------------------------
    with st.sidebar:
        # Global design system injection (inside sidebar so main content starts cleanly at top)
        render_html(build_custom_css(b64_hero))

        # Top: Clean, professional bicycle logo and branding in one horizontal row
        render_html(
            """
            <div style="display:flex; align-items:center; gap:11px; padding:4px 2px 16px 2px; border-bottom:1px solid #1E293B; margin-bottom:16px;">
                <div style="width:38px; height:38px; background:linear-gradient(135deg, #1E293B, #0F172A); border:1px solid #334155; border-radius:9px; display:flex; align-items:center; justify-content:center; color:#38BDF8; flex-shrink:0;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="5.5" cy="17.5" r="3.5"/>
                        <circle cx="18.5" cy="17.5" r="3.5"/>
                        <path d="M15 6a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm-3 11.5L9.5 9h3l2.5 5 3.5-5"/>
                        <path d="M12 9l-4 8.5"/>
                    </svg>
                </div>
                <div style="display:flex; flex-direction:column; justify-content:center; min-width:0;">
                    <div style="display:flex; align-items:center; gap:6px; line-height:1.1;">
                        <span style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.15rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.01em;">
                            SmartBike
                        </span>
                        <span style="font-size:0.62rem; font-weight:700; color:#38BDF8; background:rgba(56, 189, 248, 0.12); border:1px solid rgba(56, 189, 248, 0.28); padding:1px 5px; border-radius:4px; letter-spacing:0.04em;">
                            AI
                        </span>
                    </div>
                    <div style="font-family:'Inter',sans-serif; font-size:0.72rem; color:#64748B; margin-top:3px; font-weight:500;">
                        Urban Mobility AI
                    </div>
                </div>
            </div>
            <div style="font-size:0.68rem; font-weight:700; color:#475569; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px; padding-left:2px;">
                Platform Navigation
            </div>
            """
        )

        # Middle: Three navigation links with clean, consistent line icons
        nav_labels = {
            "Dashboard": "⊞  Dashboard",
            "Demand History": "◷  Demand History",
            "About Project": "ⓘ  About Project",
        }

        selected_page = st.radio(
            "Navigation Menu",
            options=["Dashboard", "Demand History", "About Project"],
            format_func=lambda p: nav_labels.get(p, p),
            index=0,
            label_visibility="collapsed"
        )

        # Bottom widgets: dispatch rules and footer pinned to bottom
        render_html(
            """
            <div class="sidebar-bottom-anchor" style="margin-top:auto; display:flex; flex-direction:column; gap:12px; width:100%; padding-bottom:8px;">
                <div style="padding:12px 14px; background:rgba(30, 41, 59, 0.35); border:1px solid #1E293B; border-radius:8px;">
                    <div style="display:flex; align-items:center; gap:6px; color:#38BDF8; font-size:0.74rem; font-weight:600; margin-bottom:4px;">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                        </svg>
                        Smart Dispatch
                    </div>
                    <div style="font-size:0.71rem; color:#94A3B8; line-height:1.45;">
                        Nearby donor balancing with GPS distance ranking and safety buffers.
                    </div>
                </div>

                <div style="padding-top:12px; border-top:1px solid #1E293B; display:flex; align-items:center; justify-content:space-between; font-size:0.70rem; color:#64748B;">
                    <span>SmartBike AI v1.2</span>
                    <span>SF Bay Area</span>
                </div>
            </div>
            """
        )


    # ============================================================
    # 1. DASHBOARD PAGE
    # ============================================================
    if selected_page == "Dashboard":
        # State management for predictions
        if "has_predicted" not in st.session_state:
            st.session_state["has_predicted"] = True
            st.session_state["active_stn"] = default_station
            st.session_state["active_date"] = pd.to_datetime("2015-06-15").date()
            st.session_state["active_hour_int"] = 8
            st.session_state["active_hour_label"] = "08:00 AM"

        has_predicted = st.session_state.get("has_predicted", True)

        # Reactive synchronization with interactive widgets
        if "sel_station_dropdown" in st.session_state and st.session_state["sel_station_dropdown"] in all_station_ids:
            curr_stn = st.session_state["sel_station_dropdown"]
            st.session_state["active_stn"] = curr_stn
        else:
            curr_stn = st.session_state.get("active_stn", default_station)

        min_test_date = pd.to_datetime("2015-05-14").date()
        max_test_date = pd.to_datetime("2026-12-31").date()
        if "sel_date_picker" in st.session_state and st.session_state["sel_date_picker"] is not None:
            curr_date = st.session_state["sel_date_picker"]
            if curr_date < min_test_date:
                curr_date = min_test_date
                st.session_state["sel_date_picker"] = min_test_date
            elif curr_date > max_test_date:
                curr_date = max_test_date
                st.session_state["sel_date_picker"] = max_test_date
            st.session_state["active_date"] = curr_date
        else:
            curr_date = st.session_state.get("active_date", pd.to_datetime("2015-06-15").date())
            if curr_date < min_test_date:
                curr_date = min_test_date
            elif curr_date > max_test_date:
                curr_date = max_test_date

        if "sel_hour_dropdown" in st.session_state:
            curr_hour_label = st.session_state["sel_hour_dropdown"]
            curr_hour_int = hour_to_int.get(curr_hour_label, 8)
            if curr_date == min_test_date and curr_hour_int < 2:
                curr_hour_int = 2
                curr_hour_label = hour_options[2]
                st.session_state["sel_hour_dropdown"] = curr_hour_label
            st.session_state["active_hour_int"] = curr_hour_int
            st.session_state["active_hour_label"] = curr_hour_label
        else:
            curr_hour_int = st.session_state.get("active_hour_int", 8)
            if curr_date == min_test_date and curr_hour_int < 2:
                curr_hour_int = 2
            curr_hour_label = st.session_state.get("active_hour_label", hour_options[curr_hour_int] if curr_hour_int < len(hour_options) else "08:00 AM")

        curr_target_dt = pd.Timestamp(curr_date) + pd.Timedelta(hours=curr_hour_int)

        curr_capacity = get_station_capacity(curr_stn)
        curr_available = estimate_available_bikes(curr_stn, curr_capacity)
        start_hour_str = hour_labels[curr_hour_int]
        end_hour_str = hour_labels[(curr_hour_int + 1) % 24]
        destination_station_name = station_names.get(curr_stn, f"Station {curr_stn}")

        # Model Inference & Redistribution Calculation
        if has_predicted:
            predicted_demand_val = 12.0
            predicted_departures = 12
            method_badge_label = "Verified ML Model"
            method_badge_color = "#2563EB"
            is_future_extrapolated = False

            actual_departures_obs = None
            if production_model is not None and audited_features:
                df_feat = load_feature_dataset()
                # Base prediction using normal model inference (special_event="No Event" by default)
                X_sample_base, act_val, msg = prepare_prediction_features(
                    curr_stn, curr_target_dt, df_feat, audited_features, "No Event"
                )
                if act_val is not None:
                    actual_departures_obs = int(round(float(act_val)))

                if X_sample_base is not None:
                    predicted_demand_val = make_prediction(production_model, X_sample_base)
                    predicted_departures = round_half_up(predicted_demand_val)
                    method_badge_label = "Verified ML Model"
                    method_badge_color = "#2563EB"
                    is_future_extrapolated = False
                else:
                    is_future_extrapolated = True
                    profile = get_station_hourly_profile(curr_stn)
                    if profile is not None and not profile["hourly_df"].empty:
                        h_df = profile["hourly_df"]
                        h_row = h_df[h_df["hour"] == curr_hour_int]
                        h_base = float(h_row["demand"].values[0]) if not h_row.empty else 3.0
                        is_wkday = 1 if curr_target_dt.weekday() < 5 else 0
                        day_factor = (profile["weekday_avg"] / (profile["weekday_avg"] + profile["weekend_avg"] + 1e-5) * 2.0) if is_wkday else (profile["weekend_avg"] / (profile["weekday_avg"] + profile["weekend_avg"] + 1e-5) * 2.0)
                        predicted_demand_val = max(0.0, round(h_base * day_factor, 2))
                    else:
                        stn_baseline = get_station_baseline(curr_stn, eda_data)
                        mult = 2.2 if (7 <= curr_hour_int <= 9 or 16 <= curr_hour_int <= 19) else 0.8
                        predicted_demand_val = max(0.0, round(stn_baseline * mult, 2))

                    predicted_departures = round_half_up(predicted_demand_val)
                    method_badge_label = "Future Forecast (Historical Profile)"
                    method_badge_color = "#0284C7"
            else:
                profile = get_station_hourly_profile(curr_stn)
                h_base = 3.0
                if profile is not None and not profile["hourly_df"].empty:
                    h_row = profile["hourly_df"][profile["hourly_df"]["hour"] == curr_hour_int]
                    if not h_row.empty:
                        h_base = float(h_row["demand"].values[0])
                predicted_demand_val = max(0.0, round(h_base, 2))
                predicted_departures = round_half_up(predicted_demand_val)
                method_badge_label = "Historical Profile"
                method_badge_color = "#0284C7"

            # Inventory balance & shortage calculation
            # Physical stock can never be negative
            estimated_remaining_stock = max(0, curr_available - predicted_departures)
            shortage_count = max(0, predicted_departures - curr_available)
            status_key = "shortage" if shortage_count > 0 else ("low_reserve" if (curr_available - predicted_departures) < 3 else "sufficient")

            # Nearest Suitable Donor Matching using Real Station Coordinates
            df_feat_all = load_feature_dataset()
            donor_match = find_nearest_safe_donor(
                target_station_id=curr_stn,
                target_dt=curr_target_dt,
                dest_demand=predicted_departures,
                dest_available=curr_available,
                dest_name=destination_station_name,
                df_features=df_feat_all,
                eda_station_df=eda_data.get("station"),
                safety_buffer=3,
                station_names=station_names,
            )
            donor_id = donor_match["donor_id"]
            donor_name = donor_match["donor_name"]
            if donor_id is not None and donor_name and not str(donor_name).startswith(f"Station {donor_id}"):
                donor_name = f"Station {donor_id} — {donor_name}"
            donor_distance_km = donor_match["distance_km"]
            donor_available = donor_match["donor_available"]
            donor_remaining = donor_match["donor_remaining"]
            donor_surplus = donor_match["donor_safe_surplus"]
            transfer_qty = donor_match["recommended_transfer"]
            donor_status = donor_match["status"]
            donor_is_partial = donor_match.get("is_partial", False)
            donor_is_multi_donor = donor_match.get("is_multi_donor", False)
            donor_list = donor_match.get("donors", [])
            total_transferred = donor_match.get("total_transferred", transfer_qty or 0)
            unmet_demand = donor_match.get("unmet_demand", 0)
            donor_reason = donor_match["rationale"]
        else:
            predicted_departures = None
            estimated_remaining_stock = None
            shortage_count = None
            transfer_qty = None
            status_key = None
            donor_id = None
            donor_name = None
            donor_distance_km = 0.0
            donor_available = 0
            donor_remaining = 0
            donor_surplus = 0
            donor_reason = ""
            donor_status = None
            donor_is_partial = False
            donor_is_multi_donor = False
            donor_list = []
            total_transferred = 0
            unmet_demand = 0

        # ------------------------------------------------------------
        # HERO BANNER
        # ------------------------------------------------------------
        render_html(
            f"""
            <div class="hero-banner-wrap">
                <div style="display:flex; justify-content:space-between; align-items:center; width:100%; position:relative; z-index:2;">
                    <div>
                        <div style="display:inline-flex; align-items:center; gap:6px; background:rgba(15, 23, 42, 0.60); border:1px solid rgba(255, 255, 255, 0.25); color:#FFFFFF; font-size:0.68rem; font-weight:700; padding:3px 10px; border-radius:9999px; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px; backdrop-filter:blur(6px);">
                            San Francisco Bay Area
                        </div>
                        <div class="hero-heading">
                            Bike Demand &amp; Redistribution
                        </div>
                        <div class="hero-body">
                            Plan bike availability and move inventory where riders need it most.
                        </div>
                    </div>
                    <div style="background:rgba(15, 23, 42, 0.60); backdrop-filter:blur(10px); border:1px solid rgba(255, 255, 255, 0.22); border-radius:12px; padding:8px 16px; text-align:right;">
                        <div style="font-size:0.65rem; font-weight:700; color:#93C5FD; text-transform:uppercase; letter-spacing:0.04em;">Network</div>
                        <div style="font-size:1.15rem; font-weight:800; color:#FFFFFF;">70 Stations</div>
                    </div>
                </div>
            </div>
            """
        )

        # ------------------------------------------------------------
        # ROW 1: FOUR SUMMARY CARDS
        # ------------------------------------------------------------
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4, gap="medium")

        with kpi_c1:
            if has_predicted:
                p_num = str(predicted_departures)
                p_unit = "trip" if predicted_departures == 1 else "trips"
                sub_text = f"Continuous rate: {predicted_demand_val:.2f} trips/hr"
                val_html = f'<div class="kpi-val-row"><span class="kpi-val-num">{p_num}</span><span class="kpi-val-unit">{p_unit}</span></div>'
            else:
                sub_text = "Awaiting prediction"
                val_html = '<div class="kpi-val-row"><span class="kpi-val-num">—</span></div>'

            render_html(
                f"""
                <div class="kpi-card kpi-card-blue">
                    <div class="kpi-title-row">
                        <span class="kpi-label">Predicted Demand</span>
                        <div class="kpi-icon-pill" style="background:#EFF6FF; color:#2563EB;">🚴</div>
                    </div>
                    {val_html}
                    <div class="kpi-desc">{sub_text}</div>
                </div>
                """
            )

        with kpi_c2:
            render_html(
                f"""
                <div class="kpi-card kpi-card-green">
                    <div class="kpi-title-row">
                        <span class="kpi-label">Bikes Available</span>
                        <div class="kpi-icon-pill" style="background:#ECFDF5; color:#059669;">🚲</div>
                    </div>
                    <div class="kpi-val-row">
                        <span class="kpi-val-num">{curr_available}</span>
                        <span class="kpi-val-unit">{'bike' if curr_available == 1 else 'bikes'}</span>
                    </div>
                    <div class="kpi-desc">{curr_capacity} docks total</div>
                </div>
                """
            )

        with kpi_c3:
            if has_predicted:
                shortage_num = str(shortage_count)
                shortage_unit = "bike" if shortage_count == 1 else "bikes"
                shortage_sub = "Additional bikes needed" if shortage_count > 0 else "Sufficient inventory"
                card_accent = "kpi-card-amber" if shortage_count > 0 else "kpi-card-green"
                pill_bg = "#FFFBEB" if shortage_count > 0 else "#ECFDF5"
                pill_color = "#D97706" if shortage_count > 0 else "#059669"
                pill_icon = "⚠️" if shortage_count > 0 else "✓"
                val_html = f'<div class="kpi-val-row"><span class="kpi-val-num">{shortage_num}</span><span class="kpi-val-unit">{shortage_unit}</span></div>'
            else:
                shortage_sub = "Awaiting prediction"
                card_accent = "kpi-card-amber"
                pill_bg = "#F8FAFC"
                pill_color = "#64748B"
                pill_icon = "⏱️"
                val_html = '<div class="kpi-val-row"><span class="kpi-val-num">—</span></div>'

            render_html(
                f"""
                <div class="kpi-card {card_accent}">
                    <div class="kpi-title-row">
                        <span class="kpi-label">Expected Shortage</span>
                        <div class="kpi-icon-pill" style="background:{pill_bg}; color:{pill_color};">{pill_icon}</div>
                    </div>
                    {val_html}
                    <div class="kpi-desc">{shortage_sub}</div>
                </div>
                """
            )

        with kpi_c4:
            if has_predicted:
                transfer_num = str(transfer_qty) if transfer_qty is not None else "0"
                transfer_unit = "bike" if transfer_qty == 1 else "bikes"
                transfer_sub = "Recommended to bring in" if transfer_qty and transfer_qty > 0 else "No transfer needed"
                card_accent = "kpi-card-purple" if transfer_qty and transfer_qty > 0 else "kpi-card-green"
                pill_bg = "#F5F3FF" if transfer_qty and transfer_qty > 0 else "#ECFDF5"
                pill_color = "#7C3AED" if transfer_qty and transfer_qty > 0 else "#059669"
                pill_icon = "📦" if transfer_qty and transfer_qty > 0 else "✓"
                val_html = f'<div class="kpi-val-row"><span class="kpi-val-num">{transfer_num}</span><span class="kpi-val-unit">{transfer_unit}</span></div>'
            else:
                transfer_sub = "Awaiting prediction"
                card_accent = "kpi-card-purple"
                pill_bg = "#F8FAFC"
                pill_color = "#64748B"
                pill_icon = "⏱️"
                val_html = '<div class="kpi-val-row"><span class="kpi-val-num">—</span></div>'

            render_html(
                f"""
                <div class="kpi-card {card_accent}">
                    <div class="kpi-title-row">
                        <span class="kpi-label">Suggested Transfer</span>
                        <div class="kpi-icon-pill" style="background:{pill_bg}; color:{pill_color};">{pill_icon}</div>
                    </div>
                    {val_html}
                    <div class="kpi-desc">{transfer_sub}</div>
                </div>
                """
            )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # ROW 2: PREDICTION FORM AND RESULT (EQUAL-WIDTH PANELS)
        # ------------------------------------------------------------
        col_form, col_res = st.columns([1, 1], gap="medium")

        with col_form:
            with st.container(border=True):
                render_html(
                    """
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.15rem; font-weight:800; color:#0F172A; margin-bottom:12px;">
                        Bike Demand Prediction
                    </div>
                    """
                )

                st_idx = all_station_ids.index(curr_stn) if curr_stn in all_station_ids else 0
                selected_stn = st.selectbox(
                    "Bike Station",
                    options=all_station_ids,
                    index=st_idx,
                    format_func=lambda sid: station_names.get(sid, f"Station {sid}"),
                    key="sel_station_dropdown",
                )

                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    min_test_date = pd.to_datetime("2015-05-14").date()
                    if curr_date < min_test_date:
                        curr_date = min_test_date
                    selected_date = st.date_input(
                        "Date",
                        value=curr_date,
                        min_value=min_test_date,
                        max_value=pd.to_datetime("2026-12-31").date(),
                        help="Calendar starts on May 14, 2015 (the beginning of the unseen testing dataset).",
                        key="sel_date_picker",
                    )
                with sub_c2:
                    if selected_date == min_test_date:
                        avail_hour_options = [lbl for lbl in hour_options if hour_to_int[lbl] >= 2]
                    else:
                        avail_hour_options = hour_options

                    if curr_hour_label in avail_hour_options:
                        h_idx = avail_hour_options.index(curr_hour_label)
                    elif curr_hour_int >= 2:
                        lbl_match = hour_options[curr_hour_int] if curr_hour_int < len(hour_options) else None
                        h_idx = avail_hour_options.index(lbl_match) if lbl_match in avail_hour_options else 0
                    else:
                        h_idx = 0

                    selected_hour_str = st.selectbox(
                        "Hour",
                        options=avail_hour_options,
                        index=h_idx,
                        help="On May 14, 2015, testing data begins at 02:00 AM. Prior hours (12:00 AM–01:59 AM) are excluded." if selected_date == min_test_date else None,
                        key="sel_hour_dropdown",
                    )

                chosen_hour = hour_to_int.get(selected_hour_str, 8)

                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                if st.button("Predict Demand", type="primary", use_container_width=True):
                    st.session_state["has_predicted"] = True
                    st.session_state["active_stn"] = selected_stn
                    st.session_state["active_date"] = selected_date
                    st.session_state["active_hour_int"] = chosen_hour
                    st.session_state["active_hour_label"] = selected_hour_str
                    st.rerun()

        with col_res:
            with st.container(border=True):
                render_html(
                    """
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.15rem; font-weight:800; color:#0F172A; margin-bottom:12px;">
                        Prediction Result
                    </div>
                    """
                )

                if not has_predicted:
                    render_html(
                        """
                        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:220px; text-align:center; padding: 20px 0;">
                            <div style="font-size:2.2rem; margin-bottom:8px;">⏱️</div>
                            <div style="font-size:1.0rem; font-weight:600; color:#334155; max-width:320px; line-height:1.45;">
                                Choose a station and time, then click Predict Demand.
                            </div>
                        </div>
                        """
                    )
                else:
                    # Format continuous rate with sufficient precision so small rates (e.g. 0.07, 0.09) are prominent
                    if predicted_demand_val == 0.0:
                        formatted_rate = "0.00"
                    elif predicted_demand_val < 0.01:
                        formatted_rate = f"{predicted_demand_val:.3f}"
                    else:
                        formatted_rate = f"{predicted_demand_val:.2f}"

                    # Badges accurately reflecting the continuous prediction rate and thresholds
                    if predicted_demand_val >= 15.0:
                        demand_tag = '<span style="background:#FEF2F2; color:#991B1B; border:1px solid #FECACA; font-weight:700; font-size:0.75rem; padding:3px 10px; border-radius:9999px;">High Demand</span>'
                    elif predicted_demand_val >= 5.0:
                        demand_tag = '<span style="background:#FFFBEB; color:#92400E; border:1px solid #FDE68A; font-weight:700; font-size:0.75rem; padding:3px 10px; border-radius:9999px;">Moderate Demand</span>'
                    elif predicted_demand_val > 0.0:
                        demand_tag = '<span style="background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; font-weight:700; font-size:0.75rem; padding:3px 10px; border-radius:9999px;">Low Demand</span>'
                    else:
                        demand_tag = '<span style="background:#F1F5F9; color:#475569; border:1px solid #CBD5E1; font-weight:700; font-size:0.75rem; padding:3px 10px; border-radius:9999px;">Zero Demand</span>'

                    method_badge = f'<span style="background:#F1F5F9; color:{method_badge_color}; border:1px solid #CBD5E1; font-weight:700; font-size:0.72rem; padding:3px 8px; border-radius:9999px; margin-left:6px;">{method_badge_label}</span>'
                    int_trips_unit_text = "trip" if predicted_departures == 1 else "trips"

                    render_html(
                        f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:0.80rem; font-weight:600; color:#64748B;">Selected Hour: {start_hour_str} – {end_hour_str}</span>
                            <div style="display:flex; align-items:center;">
                                {demand_tag}
                                {method_badge}
                            </div>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin:8px 0 16px 0; padding:12px 14px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;">
                            <div>
                                <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:2px;">Predicted Demand Rate</div>
                                <div style="display:flex; align-items:baseline; gap:6px;">
                                    <span style="font-family:'Plus Jakarta Sans',sans-serif; font-size:2.8rem; font-weight:800; color:#0F172A; line-height:1;">{formatted_rate}</span>
                                    <span style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.15rem; font-weight:700; color:#2563EB;">trips/hr</span>
                                </div>
                            </div>
                            <div style="text-align:right; padding-bottom:4px;">
                                <div style="font-size:0.72rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:2px;">Rounded Demand Estimate</div>
                                <div style="display:flex; align-items:baseline; justify-content:flex-end; gap:5px;">
                                    <span style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.8rem; font-weight:800; color:#1E293B; line-height:1;">{predicted_departures}</span>
                                    <span style="font-size:0.92rem; font-weight:700; color:#64748B;">{int_trips_unit_text}</span>
                                </div>
                            </div>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.80rem; color:#64748B; border-top:1px solid #F1F5F9; padding-top:10px;">
                            <span>Station Capacity: <strong>{curr_capacity} Docks</strong></span>
                            <span>Currently Docked: <strong>{curr_available} Bikes</strong></span>
                        </div>
                        """
                    )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # ROW 3: BIKE AVAILABILITY (WILL THIS STATION HAVE ENOUGH BIKES?)
        # ------------------------------------------------------------
        with st.container(border=True):
            if not has_predicted:
                render_html(
                    """
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.18rem; font-weight:800; color:#0F172A; margin-bottom:6px;">
                        Will This Station Have Enough Bikes?
                    </div>
                    <div style="font-size:0.88rem; color:#64748B; padding: 12px 0;">
                        Click Predict Demand above to check if this station will have enough bikes for the selected hour.
                    </div>
                    """
                )
            else:
                raw_remaining = curr_available - predicted_departures
                rem_val = abs(int(raw_remaining))
                rem_unit = "bike" if rem_val == 1 else "bikes"
                avail_unit = "bike" if curr_available == 1 else "bikes"
                dep_unit = "departure" if predicted_departures == 1 else "departures"

                if shortage_count > 0:
                    status_pill = '<span style="background:#FEF2F2; color:#991B1B; border:1px solid #FECACA; font-weight:700; font-size:0.78rem; padding:4px 12px; border-radius:9999px;">● More Bikes Needed</span>'
                    b_word_need = "bike" if shortage_count == 1 else "bikes"
                    explain_sentence = f"{curr_available} {avail_unit} are available, but {predicted_departures} {dep_unit} are predicted. The station may need {shortage_count} additional {b_word_need}."
                    rem_color = "#DC2626"
                else:
                    status_pill = '<span style="background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; font-weight:700; font-size:0.78rem; padding:4px 12px; border-radius:9999px;">● Sufficient Inventory</span>'
                    explain_sentence = f"{curr_available} {avail_unit} are available and {predicted_departures} {dep_unit} are predicted. The station will have {rem_val} {rem_unit} remaining."
                    rem_color = "#059669"

                render_html(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.18rem; font-weight:800; color:#0F172A;">
                            Will This Station Have Enough Bikes?
                        </div>
                        {status_pill}
                    </div>

                    <div class="avail-grid">
                        <div class="avail-box">
                            <div class="avail-box-lbl">Bikes Available</div>
                            <div class="avail-box-num">
                                <span class="avail-val" style="color:#0F172A;">{curr_available}</span>
                                <span class="avail-unit">{avail_unit}</span>
                            </div>
                        </div>
                        <div class="avail-box">
                            <div class="avail-box-lbl">Bikes Expected To Be Needed</div>
                            <div class="avail-box-num">
                                <span class="avail-val" style="color:#2563EB;">{predicted_departures}</span>
                                <span class="avail-unit">{dep_unit}</span>
                            </div>
                        </div>
                        <div class="avail-box">
                            <div class="avail-box-lbl">Estimated Bikes Remaining</div>
                            <div class="avail-box-num">
                                <span class="avail-val" style="color:{rem_color};">{rem_val}</span>
                                <span class="avail-unit">{rem_unit}</span>
                            </div>
                        </div>
                    </div>

                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px 14px; font-size:0.88rem; color:#334155; line-height:1.45; margin-top:8px;">
                        {explain_sentence}
                    </div>
                    <div style="font-size:0.75rem; color:#64748B; margin-top:8px;">
                        * Note: Estimated remaining bikes assume no incoming bikes are returned during the selected hour.
                    </div>
                    """
                )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # ROW 4: BIKE TRANSFER RECOMMENDATION
        # ------------------------------------------------------------
        with st.container(border=True):
            render_html(
                """
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.18rem; font-weight:800; color:#0F172A; margin-bottom:12px;">
                    Bike Transfer Recommendation
                </div>
                """
            )

            if not has_predicted:
                render_html(
                    """
                    <div style="font-size:0.88rem; color:#64748B; padding: 8px 0;">
                        Transfer recommendations will appear after predicting demand.
                    </div>
                    """
                )
            else:
                if shortage_count == 0 or donor_status == "enough_bikes":
                    render_html(
                        """
                        <div style="background:#ECFDF5; border:1.5px solid #A7F3D0; border-radius:12px; padding:16px 20px; color:#065F46; font-size:0.92rem; font-weight:600; display:flex; align-items:center; gap:10px;">
                            <span style="font-size:1.3rem;">✓</span>
                            <span>No transfer needed — Sufficient bikes available at this station.</span>
                        </div>
                        """
                    )
                elif not donor_list or donor_status == "no_suitable_donor" or total_transferred == 0:
                    render_html(
                        f"""
                        <div style="background:#FFFBEB; border:1.5px solid #FDE68A; border-radius:12px; padding:16px 20px; color:#92400E; font-size:0.92rem; font-weight:600; display:flex; align-items:center; gap:10px;">
                            <span style="font-size:1.3rem;">⚠️</span>
                            <span><strong>No suitable nearby donor station found.</strong> Station is expected to need {format_bikes(shortage_count)}, but no nearby station currently has surplus bikes available to share.</span>
                        </div>
                        """
                    )
                elif not donor_is_multi_donor and len(donor_list) == 1:
                    # SINGLE-DONOR RECOMMENDATION (Preserved when 1 donor covers the shortage)
                    d_single = donor_list[0]
                    dist_text = f"📍 {d_single['distance_km']} km away" if d_single['distance_km'] > 0 else "Nearby station"
                    b_word_need = "bike" if shortage_count == 1 else "bikes"
                    raw_remaining = curr_available - predicted_departures
                    rem_val = abs(int(raw_remaining))
                    donor_rem_val = abs(int(d_single["remaining"]))

                    render_html(
                        f"""
                        <div class="redistrib-flow-container">
                            <div class="redistrib-box-from">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                    <span style="font-size:0.72rem; font-weight:800; color:#065F46; text-transform:uppercase; letter-spacing:0.04em;">TAKE BIKES FROM</span>
                                    <span style="font-size:0.72rem; font-weight:700; color:#047857; background:#D1FAE5; padding:2px 8px; border-radius:9999px;">{dist_text}</span>
                                </div>
                                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.08rem; font-weight:800; color:#0F172A; margin:6px 0 4px 0;">
                                    {d_single['donor_name']}
                                </div>
                                <div style="font-size:0.78rem; color:#475569; margin:4px 0;">
                                    Available: <strong>{d_single['available']}</strong> &bull; Est. remaining: <strong>{donor_rem_val}</strong>
                                </div>
                                <div style="font-size:0.84rem; font-weight:700; color:#059669; margin-top:6px;">
                                    Bikes available to share: {d_single['safe_surplus']}
                                </div>
                            </div>

                            <div class="redistrib-box-move">
                                <div style="font-size:0.72rem; font-weight:800; color:#1E40AF; text-transform:uppercase; letter-spacing:0.04em;">MOVE</div>
                                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.30rem; font-weight:800; color:#2563EB; margin:6px 0; white-space:nowrap;">
                                    &rarr; {format_bikes(d_single['recommended_transfer'])} &rarr;
                                </div>
                                <div style="font-size:0.72rem; font-weight:600; color:#3B82F6;">Safe Transfer</div>
                            </div>

                            <div class="redistrib-box-to">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                    <span style="font-size:0.72rem; font-weight:800; color:#991B1B; text-transform:uppercase; letter-spacing:0.04em;">BRING BIKES TO</span>
                                    <span style="font-size:0.72rem; font-weight:700; color:#B91C1C; background:#FEE2E2; padding:2px 8px; border-radius:9999px;">Selected Station</span>
                                </div>
                                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.08rem; font-weight:800; color:#0F172A; margin:6px 0 4px 0;">
                                    {destination_station_name}
                                </div>
                                <div style="font-size:0.78rem; color:#475569; margin:4px 0;">
                                    Available: <strong>{curr_available}</strong> &bull; Est. remaining: <strong>{rem_val}</strong>
                                </div>
                                <div style="font-size:0.84rem; font-weight:700; color:#DC2626; margin-top:6px;">
                                    Additional bikes needed: {shortage_count}
                                </div>
                            </div>
                        </div>

                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px 16px; font-size:0.88rem; color:#334155; line-height:1.5; margin-top:12px;">
                            {d_single['donor_name']} has {d_single['safe_surplus']} bikes available to share after meeting its expected demand. Move {format_bikes(d_single['recommended_transfer'])} to {destination_station_name}, which is expected to need {shortage_count} additional {b_word_need}.
                        </div>
                        """
                    )
                else:
                    # MULTI-STATION TRANSFER PLAN
                    if unmet_demand == 0:
                        status_badge_html = '<span style="background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; font-weight:700; font-size:0.80rem; padding:4px 12px; border-radius:9999px;">✓ Destination shortage covered</span>'
                        unmet_color = "#059669"
                    else:
                        status_badge_html = f'<span style="background:#FEF2F2; color:#991B1B; border:1px solid #FECACA; font-weight:700; font-size:0.80rem; padding:4px 12px; border-radius:9999px;">⚠️ Remaining shortage: {format_bikes(unmet_demand)}</span>'
                        unmet_color = "#DC2626"

                    donors_html = []
                    for d in donor_list:
                        d_id = d["donor_id"]
                        d_name = d["donor_name"]
                        full_station_label = f"Station {d_id} — {d_name}" if not str(d_name).startswith(f"Station {d_id}") else d_name
                        d_dist = f"📍 {d['distance_km']} km away" if d["distance_km"] > 0 else "Nearby station"
                        d_avail = d["available"]
                        d_dem = d["demand"]
                        d_buf = d["safety_buffer"]
                        d_surp = d["safe_surplus"]
                        d_xfer = d["recommended_transfer"]
                        d_rem = d["remaining_after_transfer"]

                        donor_card = f"""
                        <div style="background:#FFFFFF; border:1.5px solid #E2E8F0; border-radius:12px; padding:16px 20px; margin-bottom:12px; box-shadow:0 2px 8px -1px rgba(15,23,42,0.04); transition:all 0.2s ease;">
                            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:12px;">
                                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.05rem; font-weight:800; color:#0F172A;">
                                    {full_station_label}
                                </div>
                                <span style="font-size:0.75rem; font-weight:700; color:#047857; background:#D1FAE5; padding:3px 10px; border-radius:9999px;">
                                    {d_dist}
                                </span>
                            </div>

                            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;">
                                <div style="display:grid; grid-template-columns: repeat(4, auto); gap:16px; background:#F8FAFC; border:1px solid #F1F5F9; border-radius:8px; padding:10px 16px;">
                                    <div>
                                        <div style="font-size:0.68rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:0.02em;">Available</div>
                                        <div style="font-size:0.95rem; font-weight:800; color:#0F172A;">{d_avail} <span style="font-size:0.70rem; color:#64748B; font-weight:600;">bikes</span></div>
                                    </div>
                                    <div>
                                        <div style="font-size:0.68rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:0.02em;">Expected Demand</div>
                                        <div style="font-size:0.95rem; font-weight:800; color:#2563EB;">{d_dem} <span style="font-size:0.70rem; color:#64748B; font-weight:600;">dep</span></div>
                                    </div>
                                    <div>
                                        <div style="font-size:0.68rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:0.02em;">Safety Reserve</div>
                                        <div style="font-size:0.95rem; font-weight:800; color:#64748B;">{d_buf} <span style="font-size:0.70rem; color:#64748B; font-weight:600;">bikes</span></div>
                                    </div>
                                    <div>
                                        <div style="font-size:0.68rem; color:#065F46; font-weight:700; text-transform:uppercase; letter-spacing:0.02em;">Safe Surplus</div>
                                        <div style="font-size:0.95rem; font-weight:800; color:#059669;">{d_surp} <span style="font-size:0.70rem; color:#059669; font-weight:600;">bikes</span></div>
                                    </div>
                                </div>

                                <div style="display:flex; flex-direction:column; align-items:flex-end; justify-content:center; background:#EFF6FF; border:1.5px solid #BFDBFE; border-radius:10px; padding:10px 20px; min-width:170px;">
                                    <div style="font-size:0.68rem; font-weight:800; color:#1E40AF; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:2px;">RECOMMENDED ACTION</div>
                                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.20rem; font-weight:800; color:#2563EB; white-space:nowrap;">
                                        Transfer {format_bikes(d_xfer)}
                                    </div>
                                    <div style="font-size:0.70rem; color:#64748B; margin-top:2px;">Remaining at station: {d_rem} bikes</div>
                                </div>
                            </div>
                        </div>
                        """
                        donors_html.append(donor_card)

                    rendered_donors = "".join(donors_html)

                    render_html(
                        f"""
                        <div style="background:#F8FAFC; border:1.5px solid #E2E8F0; border-radius:14px; padding:20px 22px; margin:10px 0 14px 0;">
                            <!-- Compact Header Row -->
                            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px; border-bottom:1px solid #E2E8F0; padding-bottom:12px;">
                                <div>
                                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.15rem; font-weight:800; color:#0F172A;">
                                        Multi-Station Transfer Plan
                                    </div>
                                    <div style="font-size:0.84rem; color:#475569; margin-top:3px;">
                                        Destination: <strong>{destination_station_name}</strong>
                                    </div>
                                </div>
                                <div>
                                    {status_badge_html}
                                </div>
                            </div>

                            <!-- 3-Metric Compact Summary Row -->
                            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:12px 18px; margin-bottom:16px;">
                                <div>
                                    <div style="font-size:0.68rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:0.03em;">Total Destination Shortage</div>
                                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.25rem; font-weight:800; color:#DC2626; margin-top:2px;">
                                        {format_bikes(shortage_count)}
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size:0.68rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:0.03em;">Total Recommended Transfer</div>
                                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.25rem; font-weight:800; color:#2563EB; margin-top:2px;">
                                        {format_bikes(total_transferred)}
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size:0.68rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:0.03em;">Remaining Unmet Demand</div>
                                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.25rem; font-weight:800; color:{unmet_color}; margin-top:2px;">
                                        {format_bikes(unmet_demand)}
                                    </div>
                                </div>
                            </div>

                            <!-- Individual Donor Cards (Sorted nearest first, no step numbering) -->
                            <div style="margin-bottom:14px;">
                                {rendered_donors}
                            </div>

                            <!-- Contextual Rationale Box -->
                            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:10px; padding:12px 16px; font-size:0.86rem; color:#334155; line-height:1.5;">
                                {donor_reason}
                            </div>
                        </div>
                        """
                    )

    # ============================================================
    # 2. DEMAND HISTORY PAGE
    # ============================================================
    elif selected_page == "Demand History":
        render_html(
            """
            <div style="padding: 14px 0 12px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 18px;">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.75rem; font-weight:800; color:#0F172A;">
                    Demand History
                </div>
                <div style="font-size:0.85rem; color:#64748B; margin-top:2px;">
                    Historical departure timelines and temporal volume patterns.
                </div>
            </div>
            """
        )

        # Set minimum date to 2013-01-01 so 2013 is always selectable in the year dropdown
        HIST_MIN_DATE = pd.to_datetime("2013-01-01").date()
        HIST_MAX_DATE = pd.to_datetime("2015-08-31").date()

        if "hist_persisted_station" not in st.session_state:
            st.session_state["hist_persisted_station"] = default_station
        if "hist_persisted_start_dt" not in st.session_state:
            st.session_state["hist_persisted_start_dt"] = pd.to_datetime("2015-06-01").date()
        if "hist_persisted_end_dt" not in st.session_state:
            st.session_state["hist_persisted_end_dt"] = pd.to_datetime("2015-06-30").date()
        if "hist_persisted_agg" not in st.session_state:
            st.session_state["hist_persisted_agg"] = "Daily"

        # Validate range and options constraints on persisted values
        cur_stn = st.session_state["hist_persisted_station"]
        if cur_stn not in all_station_ids:
            cur_stn = default_station
            st.session_state["hist_persisted_station"] = cur_stn

        cur_start = st.session_state["hist_persisted_start_dt"]
        if cur_start < HIST_MIN_DATE or cur_start > HIST_MAX_DATE:
            cur_start = pd.to_datetime("2015-06-01").date()
            st.session_state["hist_persisted_start_dt"] = cur_start

        cur_end = st.session_state["hist_persisted_end_dt"]
        if cur_end < HIST_MIN_DATE or cur_end > HIST_MAX_DATE:
            cur_end = pd.to_datetime("2015-06-30").date()
            st.session_state["hist_persisted_end_dt"] = cur_end

        agg_options = ["Daily", "Hourly"]
        cur_agg = st.session_state["hist_persisted_agg"]
        if cur_agg not in agg_options:
            cur_agg = "Daily"
            st.session_state["hist_persisted_agg"] = cur_agg

        def _on_hist_stn_change():
            st.session_state["hist_persisted_station"] = st.session_state["hist_stn_select"]

        def _on_hist_start_change():
            st.session_state["hist_persisted_start_dt"] = st.session_state["hist_start_dt"]

        def _on_hist_end_change():
            st.session_state["hist_persisted_end_dt"] = st.session_state["hist_end_dt"]

        def _on_hist_agg_change():
            st.session_state["hist_persisted_agg"] = st.session_state["hist_agg_select"]

        # Filters Grouped at the Top
        with st.container(border=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1, 1, 1], gap="medium")
            with f_col1:
                hist_stn = st.selectbox(
                    "Bike Station",
                    options=all_station_ids,
                    index=all_station_ids.index(cur_stn),
                    format_func=lambda s: station_names.get(s, f"Station {s}"),
                    key="hist_stn_select",
                    on_change=_on_hist_stn_change
                )
            with f_col2:
                hist_start = st.date_input(
                    "Start Date",
                    value=cur_start,
                    min_value=HIST_MIN_DATE,
                    max_value=HIST_MAX_DATE,
                    key="hist_start_dt",
                    on_change=_on_hist_start_change
                )
            with f_col3:
                hist_end = st.date_input(
                    "End Date",
                    value=cur_end,
                    min_value=HIST_MIN_DATE,
                    max_value=HIST_MAX_DATE,
                    key="hist_end_dt",
                    on_change=_on_hist_end_change
                )
            with f_col4:
                hist_agg = st.selectbox(
                    "Time Resolution",
                    options=agg_options,
                    index=agg_options.index(cur_agg),
                    key="hist_agg_select",
                    on_change=_on_hist_agg_change
                )

        # Sync persisted state with active widget return values
        st.session_state["hist_persisted_station"] = hist_stn
        st.session_state["hist_persisted_start_dt"] = hist_start
        st.session_state["hist_persisted_end_dt"] = hist_end
        st.session_state["hist_persisted_agg"] = hist_agg

        if hist_start > hist_end:
            st.error("Start date must be earlier than or equal to End date.")
        else:
            hist_res = query_station_history(
                station_id=hist_stn,
                start_dt=pd.Timestamp(hist_start),
                end_dt=pd.Timestamp(hist_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                aggregation=hist_agg
            )

            if hist_res:
                avg_metric_label = "Average Daily Departures" if hist_agg == "Daily" else "Average Hourly Departures"
                period_unit = "day" if hist_agg == "Daily" else "hour"

                # Uniform Equal Height KPI Cards (136px Exact)
                hk1, hk2, hk3, hk4 = st.columns(4, gap="medium")
                with hk1:
                    render_html(
                        f"""
                        <div class="kpi-card kpi-card-blue">
                            <div class="kpi-title-row">
                                <span class="kpi-label">Total Departures in Selected Period</span>
                                <div class="kpi-icon-pill" style="background:#EFF6FF; color:#2563EB;">🚴</div>
                            </div>
                            <div class="kpi-number">{hist_res['total_trips']:,}</div>
                            <div class="kpi-desc">Across {hist_res['total_records']} monitored hours</div>
                        </div>
                        """
                    )
                with hk2:
                    render_html(
                        f"""
                        <div class="kpi-card kpi-card-green">
                            <div class="kpi-title-row">
                                <span class="kpi-label">{avg_metric_label}</span>
                                <div class="kpi-icon-pill" style="background:#ECFDF5; color:#059669;">📊</div>
                            </div>
                            <div class="kpi-number">{hist_res['avg_departures']:.1f}</div>
                            <div class="kpi-desc">Departures per {period_unit}</div>
                        </div>
                        """
                    )
                with hk3:
                    render_html(
                        f"""
                        <div class="kpi-card kpi-card-amber">
                            <div class="kpi-title-row">
                                <span class="kpi-label">Peak Departure Hour</span>
                                <div class="kpi-icon-pill" style="background:#FFFBEB; color:#D97706;">⚡</div>
                            </div>
                            <div class="kpi-number" style="font-size:1.18rem; color:#D97706;">{hist_res['peak_hour_str']}</div>
                            <div class="kpi-desc">Highest average volume hour in period</div>
                        </div>
                        """
                    )
                with hk4:
                    wk_pct = (hist_res['weekday_trips'] / hist_res['total_trips'] * 100) if hist_res['total_trips'] > 0 else 0
                    render_html(
                        f"""
                        <div class="kpi-card kpi-card-purple">
                            <div class="kpi-title-row">
                                <span class="kpi-label">Weekday Share of Departures</span>
                                <div class="kpi-icon-pill" style="background:#F5F3FF; color:#7C3AED;">💼</div>
                            </div>
                            <div class="kpi-number" style="font-size:1.35rem; color:#7C3AED;">{wk_pct:.1f}%</div>
                            <div class="kpi-desc">{hist_res['weekday_trips']:,} weekday / {hist_res['weekend_trips']:,} weekend</div>
                        </div>
                        """
                    )

                st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

                # Prominent Historical Trend Chart
                with st.container(border=True):
                    render_html(
                        f"""
                        <div>
                            <div class="card-header-row">
                                <div class="card-title-group">
                                    <div class="card-icon-box" style="background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE;">
                                        📈
                                    </div>
                                    <div>
                                        <div class="card-title-main">Historical Departure Timeline ({hist_agg} Resolution)</div>
                                        <div class="card-title-sub">Recorded checkouts for {station_names.get(hist_stn, f"Station {hist_stn}")} between {hist_start} and {hist_end}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """
                    )
                    fig_t = go.Figure()
                    if hist_agg == "Daily":
                        fig_t.add_trace(go.Bar(
                            x=hist_res["trend_df"]["Timestamp"],
                            y=hist_res["trend_df"]["Departures"],
                            marker_color="#2563EB",
                            hovertemplate="%{x|%Y-%m-%d}: %{y} departures<extra></extra>"
                        ))
                    else:
                        fig_t.add_trace(go.Scatter(
                            x=hist_res["trend_df"]["Timestamp"],
                            y=hist_res["trend_df"]["Departures"],
                            mode="lines",
                            line=dict(color="#2563EB", width=2.5),
                            hovertemplate="%{x|%Y-%m-%d %I:%M %p}: %{y} departures<extra></extra>"
                        ))

                    fig_t.update_layout(
                        font=dict(family="Inter, sans-serif", color="#0F172A"),
                        xaxis=dict(
                            title=dict(text="Timestamp", font=dict(color="#475569", size=12)),
                            tickfont=dict(color="#64748B"),
                            gridcolor="#F1F5F9",
                            zerolinecolor="#E2E8F0"
                        ),
                        yaxis=dict(
                            title=dict(text="Departures", font=dict(color="#475569", size=12)),
                            tickfont=dict(color="#64748B"),
                            gridcolor="#F1F5F9",
                            zerolinecolor="#E2E8F0"
                        ),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=310,
                        plot_bgcolor="#FFFFFF",
                        paper_bgcolor="#FFFFFF"
                    )
                    st.plotly_chart(fig_t, width="stretch")
            else:
                st.info(
                    f"ℹ️ No recorded bike departures found for {station_names.get(hist_stn, f'Station {hist_stn}')} between {hist_start} and {hist_end}. Historical telemetry began on August 29, 2013."
                )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # Separate, well-organized section for Top Network Stations below the historical timeline
        with st.container(border=True):
            render_html(
                """
                <div>
                    <div class="card-header-row">
                        <div class="card-title-group">
                            <div class="card-icon-box" style="background: #FEF3C7; color: #D97706; border: 1px solid #FDE68A;">
                                🏆
                            </div>
                            <div>
                                <div class="card-title-main">Top Network Stations</div>
                                <div class="card-title-sub">Stations with the highest recorded departures across the network</div>
                            </div>
                        </div>
                    </div>
                </div>
                """
            )
            if "station" in eda_data and not eda_data["station"].empty:
                df_stn_raw = eda_data["station"].copy()
                # Aggregate by start_station_id to ensure clean totals even if multiple name variants exist
                df_grouped = df_stn_raw.groupby("start_station_id", as_index=False).agg({"demand": "sum"})
                df_grouped = df_grouped.sort_values("demand", ascending=False).head(10).reset_index(drop=True)
                df_grouped["Rank"] = range(1, len(df_grouped) + 1)
                df_grouped["Total Departures"] = df_grouped["demand"].apply(lambda v: f"{int(v):,}")
                df_grouped["Station Name"] = df_grouped["start_station_id"].apply(
                    lambda sid: station_names.get(int(sid), f"Station {int(sid)}")
                )
                df_grouped["Docks"] = df_grouped["start_station_id"].apply(lambda sid: get_station_capacity(int(sid)))

                st.dataframe(
                    df_grouped[["Rank", "Station Name", "Total Departures", "Docks"]],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "Station Name": st.column_config.TextColumn("Station Name", width="large"),
                        "Total Departures": st.column_config.TextColumn("Total Departures", width="medium"),
                        "Docks": st.column_config.NumberColumn("Docks", width="small"),
                    }
                )

    # ============================================================
    # 3. ABOUT PROJECT PAGE
    # ============================================================
    elif selected_page == "About Project":
        render_html(
            """
            <div style="padding: 14px 0 14px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 20px;">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.75rem; font-weight:800; color:#0F172A; letter-spacing:-0.02em;">
                    About SmartBike AI
                </div>
                <div style="font-size:0.88rem; color:#64748B; margin-top:3px;">
                    How the system predicts demand and recommends bike transfers.
                </div>
            </div>

            <div class="about-grid">
                <!-- Card 1: Project Overview -->
                <div class="about-card about-card-blue">
                    <div>
                        <div class="about-header-row">
                            <div class="about-icon-box" style="background:#EFF6FF; color:#2563EB; border:1px solid #BFDBFE;">
                                🏙️
                            </div>
                            <div>
                                <div class="about-card-title">Project Overview</div>
                                <div style="font-size:0.75rem; color:#64748B; font-weight:500;">Predictive mobility platform</div>
                            </div>
                        </div>
                        <div class="about-card-body">
                            SmartBike AI predicts hourly bicycle departure demand at bike stations using historical bike-sharing data. Its purpose is to help identify stations that may need additional bikes and support proactive rebalancing before docks empty during peak transit hours.
                        </div>
                    </div>
                    <div class="about-footer-tag">
                        <span style="color:#2563EB; font-weight:700;">●</span>
                        <span>Focus Corridor: San Francisco Bay Area Transit Nodes</span>
                    </div>
                </div>

                <!-- Card 2: Demand Prediction -->
                <div class="about-card about-card-purple">
                    <div>
                        <div class="about-header-row">
                            <div class="about-icon-box" style="background:#F5F3FF; color:#7C3AED; border:1px solid #DDD6FE;">
                                🧠
                            </div>
                            <div>
                                <div class="about-card-title">Demand Prediction</div>
                                <div style="font-size:0.75rem; color:#64748B; font-weight:500;">Machine learning inference workflow</div>
                            </div>
                        </div>
                        <div class="about-card-body">
                            <ul class="about-list">
                                <li>Select a station, date, and hour.</li>
                                <li>The trained machine-learning model predicts expected departures.</li>
                                <li>Compare predicted departures with estimated available bikes.</li>
                                <li>Identify potential shortages before the selected hour.</li>
                            </ul>
                        </div>
                    </div>
                    <div class="about-footer-tag">
                        <span style="color:#7C3AED; font-weight:700;">●</span>
                        <span>Model: Tuned XGBoost Regression</span>
                    </div>
                </div>

                <!-- Card 3: Bike Transfer Recommendations -->
                <div class="about-card about-card-emerald">
                    <div>
                        <div class="about-header-row">
                            <div class="about-icon-box" style="background:#ECFDF5; color:#059669; border:1px solid #A7F3D0;">
                                🔄
                            </div>
                            <div>
                                <div class="about-card-title">Bike Transfer Recommendations</div>
                                <div style="font-size:0.75rem; color:#64748B; font-weight:500;">Safe donor rebalancing logic</div>
                            </div>
                        </div>
                        <div class="about-card-body">
                            <ul class="about-list">
                                <li>Find a station that may need additional bikes.</li>
                                <li>Check nearby stations for safe surplus bikes.</li>
                                <li>Recommend a transfer only when a suitable donor exists.</li>
                                <li>Limit the transfer to the destination's shortage and each donor's safe surplus.</li>
                                <li>Multi-Station Rebalancing: When one donor cannot cover the full shortage, sequence transfers from the nearest eligible donors.</li>
                            </ul>
                            <div class="about-note-box">
                                <strong>Note:</strong> Availability and remaining-bike values are estimates based on historical network patterns and the operational assumption of zero incoming returns during the forecast hour.
                            </div>
                        </div>
                    </div>
                    <div class="about-footer-tag">
                        <span style="color:#059669; font-weight:700;">●</span>
                        <span>Inventory Rule: Donor 3-Bike Safety Buffer Preserved</span>
                    </div>
                </div>

                <!-- Card 4: Dataset and Technologies -->
                <div class="about-card about-card-cyan">
                    <div>
                        <div class="about-header-row">
                            <div class="about-icon-box" style="background:#F0F9FF; color:#0284C7; border:1px solid #BAE6FD;">
                                💻
                            </div>
                            <div>
                                <div class="about-card-title">Dataset and Technologies</div>
                                <div style="font-size:0.75rem; color:#64748B; font-weight:500;">Verified technical foundation</div>
                            </div>
                        </div>
                        <div class="about-card-body">
                            <div class="about-chip-grid">
                                <div class="about-chip">
                                    <span class="about-chip-lbl">Dataset</span>
                                    <span class="about-chip-val">SF Bay Area Bike Share</span>
                                </div>
                                <div class="about-chip">
                                    <span class="about-chip-lbl">Historical Trips</span>
                                    <span class="about-chip-val">669,959 Trips</span>
                                </div>
                                <div class="about-chip">
                                    <span class="about-chip-lbl">Stations</span>
                                    <span class="about-chip-val">70 Stations</span>
                                </div>
                                <div class="about-chip">
                                    <span class="about-chip-lbl">Production Model</span>
                                    <span class="about-chip-val">XGBoost Regressor</span>
                                </div>
                            </div>
                            <div style="margin-top:10px;">
                                <div style="font-size:0.70rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px;">Technologies Used</div>
                                <div class="tech-tag-row">
                                    <span class="tech-tag">Python 3.14</span>
                                    <span class="tech-tag">Streamlit</span>
                                    <span class="tech-tag">Pandas</span>
                                    <span class="tech-tag">NumPy</span>
                                    <span class="tech-tag">Scikit-Learn</span>
                                    <span class="tech-tag">Plotly</span>
                                    <span class="tech-tag">Apache Parquet</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="about-footer-tag">
                        <span style="color:#0284C7; font-weight:700;">●</span>
                        <span>Storage: Parquet Columnar Fast Retrieval</span>
                    </div>
                </div>
            </div>
            """
        )


if __name__ == "__main__":
    main()
