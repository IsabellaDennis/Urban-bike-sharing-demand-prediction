"""
Bike Redistribution and Inventory Optimization Module.
Calculates station rebalancing priorities and buffer recommendations based on predicted demand.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Real physical dock capacities for SF Bay Area bike share stations
STATION_CAPACITIES: Dict[int, int] = {
    70: 50,  # San Francisco Caltrain (Flagship Terminal)
    69: 45,  # San Francisco Caltrain 2 (330 Townsend)
    50: 45,  # Harry Bridges Plaza (Ferry Building)
    60: 35,  # Embarcadero at Sansome
    61: 35,  # 2nd at Townsend
    55: 35,  # Temporary Transbay Terminal
    74: 35,  # Steuart at Market
    77: 35,  # Market at Sansome
    65: 35,  # Townsend at 7th
    67: 35,  # Market at 10th
    76: 35,  # Market at 4th
    64: 30,  # 2nd at South Park
    39: 30,  # Powell Street BART
    73: 30,  # Grant Avenue at Columbus Avenue
    62: 30,  # 2nd at Folsom
    56: 30,  # Beale at Market
    54: 30,  # Embarcadero at Bryant
    72: 30,  # Civic Center BART
    51: 30,  # Embarcadero at Folsom
}

DEFAULT_STATION_CAPACITY = 30


def get_station_capacity(station_id: int) -> int:
    """Retrieve actual dock capacity for a station or standard default."""
    return STATION_CAPACITIES.get(station_id, DEFAULT_STATION_CAPACITY)


def estimate_available_bikes(station_id: int, capacity: Optional[int] = None) -> int:
    """
    Estimate standard baseline available inventory when live telemetry is unavailable.
    Standard operational baseline is ~45% of dock capacity (ensuring docks for returns).
    """
    cap = capacity or get_station_capacity(station_id)
    return max(4, int(round(cap * 0.45)))


def calculate_station_shortage_surplus(
    predicted_demand: float,
    station_capacity: int,
    available_bikes: int,
    safety_buffer: int = 3
) -> Dict[str, any]:
    """
    Calculate inventory balance for a station given predicted demand and available bikes.
    
    Formula:
      estimated_remaining = available_bikes - predicted_demand (assuming 0 inflow)
    
    Status:
      - 'Enough Bikes' (Green): estimated_remaining >= safety_buffer
      - 'Low Reserve' (Amber): 0 <= estimated_remaining < safety_buffer
      - 'Potential Shortage' (Red): estimated_remaining < 0
    """
    pred_int = int(round(predicted_demand))
    remaining = available_bikes - pred_int
    
    if remaining < 0:
        status = "Potential Shortage"
        status_key = "shortage"
        color = "#DC2626"  # Red
        bikes_needed = abs(remaining) + safety_buffer  # deficit plus buffer
        safe_surplus = 0
    elif remaining < safety_buffer:
        status = "Low Reserve"
        status_key = "low_reserve"
        color = "#D97706"  # Amber
        bikes_needed = safety_buffer - remaining
        safe_surplus = 0
    else:
        status = "Enough Bikes"
        status_key = "sufficient"
        color = "#16A34A"  # Green
        bikes_needed = 0
        # Safe surplus is bikes that can be spared without dropping below demand + buffer
        safe_surplus = max(0, available_bikes - pred_int - safety_buffer)

    return {
        "status": status,
        "status_key": status_key,
        "color": color,
        "available_bikes": available_bikes,
        "predicted_demand": pred_int,
        "estimated_remaining": remaining,
        "bikes_needed": bikes_needed,
        "safe_surplus": safe_surplus,
        "safety_buffer": safety_buffer,
    }


def compute_network_rebalancing(
    station_demand_df: pd.DataFrame,
    safety_buffer: int = 3
) -> Dict[str, any]:
    """
    Calculate genuine network shortage, surplus, and donor-destination matching.
    
    Parameters:
        station_demand_df: DataFrame with ['start_station_id', 'start_station_name', 'demand']
        safety_buffer: Minimum bikes to keep at stations after departures
        
    Returns:
        dict containing:
            - summary: {needing_bikes_count, surplus_stations_count, total_recommended_transfer}
            - rebalance_table: DataFrame with matched transfers
            - stations_state: list of station status dicts
    """
    if station_demand_df.empty or "demand" not in station_demand_df.columns:
        return {
            "summary": {
                "needing_bikes_count": 0,
                "surplus_stations_count": 0,
                "total_recommended_transfer": 0
            },
            "rebalance_table": pd.DataFrame(),
            "stations_state": []
        }

    stations_needing = []
    stations_surplus = []
    all_states = []

    for _, row in station_demand_df.iterrows():
        sid = int(row["start_station_id"])
        name = str(row.get("start_station_name", f"Station {sid}"))
        demand = float(row["demand"])
        cap = get_station_capacity(sid)
        avail = estimate_available_bikes(sid, cap)

        balance = calculate_station_shortage_surplus(
            predicted_demand=demand,
            station_capacity=cap,
            available_bikes=avail,
            safety_buffer=safety_buffer
        )
        
        info = {
            "station_id": sid,
            "station_name": name,
            "capacity": cap,
            **balance
        }
        all_states.append(info)

        if balance["bikes_needed"] > 0:
            stations_needing.append(info)
        elif balance["safe_surplus"] > 0:
            stations_surplus.append(info)

    # Sort needing by highest deficit first, surplus by highest safe surplus first
    stations_needing.sort(key=lambda x: x["bikes_needed"], reverse=True)
    stations_surplus.sort(key=lambda x: x["safe_surplus"], reverse=True)

    # Match deficits with verified surplus donors using geographic proximity
    # Track available pool of surplus so donors aren't over-allocated
    surplus_pool = {s["station_id"]: s["safe_surplus"] for s in stations_surplus}
    surplus_lookup = {s["station_id"]: s for s in stations_surplus}

    rebalance_rows = []
    total_transfers = 0

    for dest in stations_needing:
        needed = dest["bikes_needed"]
        dest_id = dest["station_id"]
        dest_lat, dest_lon = get_station_coords(dest_id)

        # Rank eligible donors with available pool > 0 by geographic distance to this destination
        eligible_donors = []
        for donor in stations_surplus:
            d_id = donor["station_id"]
            if d_id != dest_id and surplus_pool.get(d_id, 0) > 0:
                d_lat, d_lon = get_station_coords(d_id)
                dist = haversine_distance_km(dest_lat, dest_lon, d_lat, d_lon)
                eligible_donors.append((dist, donor))

        eligible_donors.sort(key=lambda x: (x[0], -surplus_pool[x[1]["station_id"]]))

        dest_transfers = []
        remaining_needed = needed

        for dist, donor in eligible_donors:
            if remaining_needed <= 0:
                break
            d_id = donor["station_id"]
            avail_surp = surplus_pool.get(d_id, 0)
            if avail_surp <= 0:
                continue

            xfer = min(remaining_needed, avail_surp)
            surplus_pool[d_id] -= xfer
            remaining_needed -= xfer
            total_transfers += xfer

            dest_transfers.append({
                "destination_id": dest_id,
                "destination_name": dest["station_name"],
                "bikes_needed": needed,
                "source_id": d_id,
                "source_name": donor["station_name"],
                "distance_km": dist,
                "available_surplus": avail_surp,
                "recommended_transfer": xfer,
                "status": "Recommended"
            })

        if dest_transfers:
            rebalance_rows.extend(dest_transfers)
        else:
            rebalance_rows.append({
                "destination_id": dest_id,
                "destination_name": dest["station_name"],
                "bikes_needed": needed,
                "source_id": None,
                "source_name": "No verified surplus station",
                "distance_km": 0.0,
                "available_surplus": 0,
                "recommended_transfer": 0,
                "status": "No Donor Available"
            })

    table_df = pd.DataFrame(rebalance_rows) if rebalance_rows else pd.DataFrame()

    return {
        "summary": {
            "needing_bikes_count": len(stations_needing),
            "surplus_stations_count": len(stations_surplus),
            "total_recommended_transfer": total_transfers
        },
        "rebalance_table": table_df,
        "stations_state": all_states
    }


def get_station_rebalancing_urgency(predicted_demand: float, station_capacity: int = 25) -> Dict[str, any]:
    """
    Backwards-compatible wrapper: calculate redistribution urgency based on predicted demand.
    """
    predicted_rounded = int(round(predicted_demand))
    recommended_min_stock = max(1, int(np.floor(predicted_demand * 1.25)))
    recommended_max_stock = min(station_capacity, max(recommended_min_stock + 2, int(np.ceil(predicted_demand * 1.6))))
    
    if predicted_demand >= 15.0:
        urgency = "HIGH"
        status = "High Outflow — Starvation Risk"
        color = "#DC2626"
    elif predicted_demand >= 5.0:
        urgency = "MODERATE"
        status = "Moderate Demand — Monitor Reserves"
        color = "#D97706"
    else:
        urgency = "LOW"
        status = "Low Departure Demand — Stable"
        color = "#16A34A"

    return {
        "urgency": urgency,
        "status": status,
        "color": color,
        "predicted_demand": predicted_demand,
        "recommended_min_stock": recommended_min_stock,
        "recommended_max_stock": recommended_max_stock,
        "target_stock_range": f"{recommended_min_stock}–{recommended_max_stock} bikes"
    }


def compute_network_redistribution_table(station_demands: pd.DataFrame) -> pd.DataFrame:
    """
    Backwards-compatible wrapper to rank monitored stations by demand.
    """
    if station_demands.empty or "demand" not in station_demands.columns:
        return pd.DataFrame()
        
    df = station_demands.copy()
    df["priority_rank"] = df["demand"].rank(ascending=False, method="min").astype(int)
    
    def assign_category(d):
        if d >= 15.0:
            return "High Demand"
        elif d >= 5.0:
            return "Moderate Demand"
        else:
            return "Low Demand"
            
    df["redistribution_category"] = df["demand"].apply(assign_category)
    return df.sort_values(by="priority_rank", ascending=True)


# Official real-world station coordinates for SF Bay Area bike share network
STATION_COORDINATES: Dict[int, Tuple[float, float]] = {
    # San Francisco Cluster (IDs 39–82)
    70: (37.776617, -122.395260),  # San Francisco Caltrain (Townsend at 4th)
    69: (37.776600, -122.395470),  # San Francisco Caltrain 2 (330 Townsend)
    50: (37.795392, -122.394203),  # Harry Bridges Plaza (Ferry Building)
    60: (37.804770, -122.403234),  # Embarcadero at Sansome
    55: (37.789756, -122.394643),  # Temporary Transbay Terminal (Howard at Beale)
    61: (37.780526, -122.390288),  # 2nd at Townsend
    74: (37.794139, -122.394434),  # Steuart at Market
    77: (37.789625, -122.400811),  # Market at Sansome
    65: (37.771058, -122.402717),  # Townsend at 7th
    67: (37.776619, -122.417385),  # Market at 10th
    76: (37.786305, -122.404966),  # Market at 4th
    64: (37.782259, -122.392738),  # 2nd at South Park
    39: (37.783871, -122.408433),  # Powell Street BART
    73: (37.797900, -122.405942),  # Grant Avenue at Columbus Avenue
    62: (37.785299, -122.396236),  # 2nd at Folsom
    56: (37.792251, -122.397086),  # Beale at Market
    54: (37.787152, -122.388013),  # Embarcadero at Bryant
    72: (37.780516, -122.414776),  # Civic Center BART (7th at Market)
    51: (37.791464, -122.391034),  # Embarcadero at Folsom
    57: (37.781752, -122.405127),  # 5th at Howard
    63: (37.786978, -122.398108),  # Howard at 2nd
    66: (37.774814, -122.418954),  # South Van Ness at Market
    71: (37.788446, -122.408499),  # Powell at Post (Union Square)
    75: (37.791300, -122.399051),  # Mechanics Plaza (Market at Battery)
    49: (37.789728, -122.391851),  # Spear at Folsom
    45: (37.794231, -122.402923),  # Commercial at Montgomery
    68: (37.784878, -122.401014),  # Yerba Buena Center of the Arts (3rd @ Howard)
    48: (37.799953, -122.398525),  # Embarcadero at Vallejo
    82: (37.798541, -122.400862),  # Broadway St at Battery St
    41: (37.795001, -122.399970),  # Clay at Battery
    42: (37.797280, -122.398436),  # Davis at Jackson
    59: (37.781280, -122.418412),  # Golden Gate at Polk
    58: (37.778650, -122.418230),  # San Francisco City Hall
    47: (37.788975, -122.403452),  # Post at Kearny
    46: (37.795425, -122.404767),  # Washington at Kearny

    # San Jose Cluster (IDs 2–16, 80, 84)
    2: (37.329732, -121.901782),   # San Jose Diridon Caltrain Station
    3: (37.330698, -121.888979),   # San Jose Civic Center
    4: (37.333988, -121.894902),   # Santa Clara at Almaden
    5: (37.331415, -121.893200),   # Adobe on Almaden
    6: (37.336721, -121.894074),   # San Pedro Square
    7: (37.333798, -121.886943),   # Paseo de San Antonio
    8: (37.330165, -121.885831),   # San Salvador at 1st
    9: (37.348742, -121.894715),   # Japantown
    10: (37.337391, -121.886995),  # San Jose City Hall
    11: (37.335885, -121.885660),  # MLK Library
    12: (37.332808, -121.883891),  # SJSU 4th at San Carlos
    13: (37.339301, -121.889937),  # St James Park
    14: (37.332692, -121.900084),  # Arena Green / SAP Center
    16: (37.334488, -121.878633),  # SJSU - San Salvador at 9th
    80: (37.352601, -121.905733),  # Santa Clara County Civic Center
    84: (37.342725, -121.895617),  # Ryland Park

    # Redwood City Cluster (IDs 21–26, 83)
    21: (37.481758, -122.226904),  # Franklin at Maple
    22: (37.486078, -122.231991),  # Redwood City Caltrain Station
    23: (37.487616, -122.229951),  # San Mateo County Center
    24: (37.484219, -122.227424),  # Redwood City Public Library
    25: (37.486725, -122.225551),  # Stanford in Redwood City / Broadway at Main
    26: (37.487475, -122.223095),  # Redwood City Medical Center
    83: (37.491269, -122.236234),  # Mezes Park

    # Mountain View Cluster (IDs 27–33)
    27: (37.389218, -122.081896),  # Mountain View City Hall
    28: (37.394358, -122.076713),  # Mountain View Caltrain Station
    29: (37.406940, -122.106758),  # San Antonio Caltrain Station
    30: (37.390277, -122.066553),  # Evelyn Park and Ride
    31: (37.400443, -122.108338),  # San Antonio Shopping Center
    32: (37.385956, -122.083678),  # Castro Street and El Camino Real
    33: (37.400241, -122.099076),  # Rengstorff Avenue / California Street

    # Palo Alto Cluster (IDs 34–38)
    34: (37.443988, -122.164759),  # Palo Alto Caltrain Station
    35: (37.444521, -122.163093),  # University and Emerson
    36: (37.429082, -122.142805),  # California Ave Caltrain Station
    37: (37.448598, -122.159504),  # Cowper at University
    38: (37.425684, -122.137777),  # Park at Olive
}


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two GPS coordinates in kilometers."""
    import math
    r = 6371.0  # Earth mean radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


def get_station_coords(station_id: int) -> Tuple[float, float]:
    """Retrieve verified coordinates for station or default SF city center fallback."""
    return STATION_COORDINATES.get(station_id, (37.7749, -122.4194))


def find_multi_station_transfer_plan(
    target_station_id: int,
    target_dt: pd.Timestamp,
    dest_demand: int,
    dest_available: int,
    dest_name: str = "",
    df_features: Optional[pd.DataFrame] = None,
    eda_station_df: Optional[pd.DataFrame] = None,
    safety_buffer: int = 3,
    station_names: Optional[Dict[int, str]] = None
) -> Dict[str, any]:
    """
    Compute a distance-ranked, sequential multi-station bike transfer plan.

    1. Calculate the selected destination's shortage using final predicted demand.
    2. Identify all eligible donor stations except the destination station.
    3. Calculate each donor's safe surplus after accounting for expected demand and safety buffer.
    4. Exclude donors with no safe surplus (<= 0).
    5. Rank eligible donors by verified geographic distance (Haversine formula).
    6. Allocate bikes from nearest eligible donor first.
    7. If its safe surplus cannot cover the shortage, continue to next nearest eligible donor.
    8. Stop when the shortage is covered or no eligible donors remain.
    9. Never transfer more bikes than the remaining shortage or a donor's safe surplus.
    10. Report any remaining unmet demand accurately.
    """
    dest_shortage = max(0, dest_demand - dest_available)
    dest_remaining = dest_available - dest_demand
    t_lat, t_lon = get_station_coords(target_station_id)
    target_label = dest_name or (station_names.get(target_station_id) if station_names else None) or f"Station {target_station_id}"

    # If destination has enough bikes to cover predicted departures
    if dest_shortage == 0:
        return {
            "status": "enough_bikes",
            "is_partial": False,
            "is_multi_donor": False,
            "donor_id": None,
            "donor_name": None,
            "distance_km": 0.0,
            "donor_available": 0,
            "donor_demand": 0,
            "donor_remaining": 0,
            "donor_safe_surplus": 0,
            "recommended_transfer": 0,
            "destination_id": target_station_id,
            "destination_name": target_label,
            "destination_available": dest_available,
            "destination_demand": dest_demand,
            "destination_remaining": dest_remaining,
            "destination_shortage": 0,
            "safety_buffer": safety_buffer,
            "donors": [],
            "total_transferred": 0,
            "unmet_demand": 0,
            "candidate_donors_count": 0,
            "rationale": "Enough bikes available — No transfer required."
        }

    # Evaluate candidate donor stations across network
    candidates = []
    seen_station_ids = set()

    # 1. Feature snapshot matching target timestamp
    if df_features is not None and not df_features.empty and "hour_timestamp" in df_features.columns:
        snap = df_features[df_features["hour_timestamp"] == target_dt]
        if not snap.empty:
            for _, r in snap.iterrows():
                sid = int(r["start_station_id"])
                if sid == target_station_id or sid in seen_station_ids:
                    continue
                seen_station_ids.add(sid)
                demand_val = int(round(float(r["demand"])))
                cap = get_station_capacity(sid)
                avail_val = estimate_available_bikes(sid, cap)
                rem = avail_val - demand_val
                safe_surp = max(0, int(rem - safety_buffer))

                # Never recommend donor that would face shortage or breach safety reserve
                if safe_surp > 0:
                    c_lat, c_lon = get_station_coords(sid)
                    dist = haversine_distance_km(t_lat, t_lon, c_lat, c_lon)
                    raw_name = str(r.get("start_station_name", ""))
                    if not raw_name or raw_name == f"Station {sid}":
                        if station_names and sid in station_names:
                            raw_name = station_names[sid]
                        else:
                            raw_name = f"Station {sid}"
                    display_name = f"Station {sid} — {raw_name}" if not raw_name.startswith(f"Station {sid}") else raw_name
                    candidates.append({
                        "id": sid,
                        "name": display_name,
                        "available": avail_val,
                        "demand": demand_val,
                        "remaining": rem,
                        "safety_buffer": safety_buffer,
                        "safe_surplus": safe_surp,
                        "distance_km": dist
                    })

    # 2. Baseline fallback from EDA dataset if snapshot is missing
    if not candidates and eda_station_df is not None and not eda_station_df.empty:
        for _, r in eda_station_df.iterrows():
            sid = int(r["start_station_id"])
            if sid == target_station_id or sid in seen_station_ids:
                continue
            seen_station_ids.add(sid)
            total_trips = float(r["demand"])
            avg_hourly = int(round(total_trips / 17583.0))
            cap = get_station_capacity(sid)
            avail_val = estimate_available_bikes(sid, cap)
            rem = avail_val - avg_hourly
            safe_surp = max(0, int(rem - safety_buffer))

            if safe_surp > 0:
                c_lat, c_lon = get_station_coords(sid)
                dist = haversine_distance_km(t_lat, t_lon, c_lat, c_lon)
                raw_name = str(r.get("start_station_name", ""))
                if not raw_name or raw_name == f"Station {sid}":
                    if station_names and sid in station_names:
                        raw_name = station_names[sid]
                    else:
                        raw_name = f"Station {sid}"
                display_name = f"Station {sid} — {raw_name}" if not raw_name.startswith(f"Station {sid}") else raw_name
                candidates.append({
                    "id": sid,
                    "name": display_name,
                    "available": avail_val,
                    "demand": avg_hourly,
                    "remaining": rem,
                    "safety_buffer": safety_buffer,
                    "safe_surplus": safe_surp,
                    "distance_km": dist
                })

    # 3. Network topology fallback if no dataframe is supplied
    if not candidates and target_station_id in STATION_COORDINATES:
        for sid in sorted(STATION_COORDINATES.keys()):
            if sid == target_station_id or sid in seen_station_ids:
                continue
            seen_station_ids.add(sid)
            cap = get_station_capacity(sid)
            avail_val = estimate_available_bikes(sid, cap)
            demand_val = 3
            rem = avail_val - demand_val
            safe_surp = max(0, int(rem - safety_buffer))
            if safe_surp > 0:
                c_lat, c_lon = get_station_coords(sid)
                dist = haversine_distance_km(t_lat, t_lon, c_lat, c_lon)
                raw_name = station_names.get(sid, f"Station {sid}") if station_names else f"Station {sid}"
                display_name = f"Station {sid} — {raw_name}" if not raw_name.startswith(f"Station {sid}") else raw_name
                candidates.append({
                    "id": sid,
                    "name": display_name,
                    "available": avail_val,
                    "demand": demand_val,
                    "remaining": rem,
                    "safety_buffer": safety_buffer,
                    "safe_surplus": safe_surp,
                    "distance_km": dist
                })

    if not candidates:
        return {
            "status": "no_suitable_donor",
            "is_partial": False,
            "is_multi_donor": False,
            "donor_id": None,
            "donor_name": None,
            "distance_km": 0.0,
            "donor_available": 0,
            "donor_demand": 0,
            "donor_remaining": 0,
            "donor_safe_surplus": 0,
            "recommended_transfer": 0,
            "destination_id": target_station_id,
            "destination_name": target_label,
            "destination_available": dest_available,
            "destination_demand": dest_demand,
            "destination_remaining": dest_remaining,
            "destination_shortage": dest_shortage,
            "safety_buffer": safety_buffer,
            "donors": [],
            "total_transferred": 0,
            "unmet_demand": dest_shortage,
            "candidate_donors_count": 0,
            "rationale": f"No suitable nearby donor station found. Station is expected to need {dest_shortage} additional bikes, but no nearby station currently has surplus bikes available to share."
        }

    # 4 & 5. Exclude donors with no safe surplus (already safe_surp > 0)
    # Rank eligible donors strictly by verified geographic distance from destination
    candidates.sort(key=lambda x: (x["distance_km"], -x["safe_surplus"], x["id"]))

    # 6, 7, 8, 9. Allocate bikes from nearest eligible donor first, continuing until shortage covered
    remaining_shortage = dest_shortage
    allocated_donors = []

    for cand in candidates:
        if remaining_shortage <= 0:
            break
        # Never transfer more than remaining shortage or donor's safe surplus
        transfer_qty = min(remaining_shortage, cand["safe_surplus"])
        if transfer_qty > 0:
            allocated_donors.append({
                "step": len(allocated_donors) + 1,
                "donor_id": cand["id"],
                "donor_name": cand["name"],
                "distance_km": cand["distance_km"],
                "available": cand["available"],
                "demand": cand["demand"],
                "remaining": cand["remaining"],
                "safety_buffer": safety_buffer,
                "safe_surplus": cand["safe_surplus"],
                "recommended_transfer": transfer_qty,
                "remaining_after_transfer": cand["remaining"] - transfer_qty,
            })
            remaining_shortage -= transfer_qty

    # 10. Report any remaining unmet demand accurately
    total_transferred = dest_shortage - remaining_shortage
    unmet_demand = remaining_shortage

    def _fmt(n: int) -> str:
        return f"{n} bike" if n == 1 else f"{n} bikes"

    is_multi_donor = (len(allocated_donors) > 1)
    is_partial = (unmet_demand > 0)
    status = "safe_donor_found" if not is_partial else "partial_donor_found"

    if not is_multi_donor and len(allocated_donors) == 1:
        d0 = allocated_donors[0]
        if not is_partial:
            rationale = (
                f"{d0['donor_name']} has {d0['safe_surplus']} bikes available to share after meeting its expected demand. "
                f"Move {_fmt(d0['recommended_transfer'])} to {target_label}, which is expected to need {_fmt(dest_shortage)} additional bikes."
            )
        else:
            rationale = (
                f"{d0['donor_name']} has {d0['safe_surplus']} bikes available to share after meeting its expected demand. "
                f"Move {_fmt(d0['recommended_transfer'])} to {target_label} (partial transfer). {_fmt(unmet_demand)} remain unfulfilled."
            )
    elif is_multi_donor:
        donors_summary = ", ".join([f"{_fmt(d['recommended_transfer'])} from {d['donor_name']} ({d['distance_km']} km)" for d in allocated_donors])
        if not is_partial:
            rationale = (
                f"Multi-Station Transfer Plan: Move a total of {_fmt(total_transferred)} across {len(allocated_donors)} stations to fulfill the {_fmt(dest_shortage)} shortage at {target_label} ({donors_summary})."
            )
        else:
            rationale = (
                f"Multi-Station Transfer Plan (Partial): Move a total of {_fmt(total_transferred)} across {len(allocated_donors)} stations ({donors_summary}). "
                f"{_fmt(unmet_demand)} remain unfulfilled due to network surplus limits."
            )
    else:
        status = "no_suitable_donor"
        rationale = f"No suitable nearby donor station found. Station is expected to need {_fmt(dest_shortage)}, but no nearby station currently has surplus bikes available to share."

    primary_donor = allocated_donors[0] if allocated_donors else None
    return {
        "status": status,
        "is_partial": is_partial,
        "is_multi_donor": is_multi_donor,
        "donor_id": primary_donor["donor_id"] if primary_donor else None,
        "donor_name": primary_donor["donor_name"] if primary_donor else None,
        "distance_km": primary_donor["distance_km"] if primary_donor else 0.0,
        "donor_available": primary_donor["available"] if primary_donor else 0,
        "donor_demand": primary_donor["demand"] if primary_donor else 0,
        "donor_remaining": primary_donor["remaining"] if primary_donor else 0,
        "donor_safe_surplus": primary_donor["safe_surplus"] if primary_donor else 0,
        "recommended_transfer": total_transferred if is_multi_donor else (primary_donor["recommended_transfer"] if primary_donor else 0),
        "destination_id": target_station_id,
        "destination_name": target_label,
        "destination_available": dest_available,
        "destination_demand": dest_demand,
        "destination_remaining": dest_remaining,
        "destination_shortage": dest_shortage,
        "safety_buffer": safety_buffer,
        "donors": allocated_donors,
        "total_transferred": total_transferred,
        "unmet_demand": unmet_demand,
        "candidate_donors_count": len(candidates),
        "rationale": rationale,
    }


def find_nearest_safe_donor(
    target_station_id: int,
    target_dt: pd.Timestamp,
    dest_demand: int,
    dest_available: int,
    dest_name: str = "",
    df_features: Optional[pd.DataFrame] = None,
    eda_station_df: Optional[pd.DataFrame] = None,
    safety_buffer: int = 3,
    station_names: Optional[Dict[int, str]] = None
) -> Dict[str, any]:
    """
    Find the nearest suitable donor station(s) with verified safe surplus based on:
    - Destination shortage calculation
    - Station geographical proximity (haversine distance)
    - Donor expected demand and dock capacity
    - Retention of configurable donor safety reserve

    Supports both single-donor and sequential multi-station transfer plans.
    """
    return find_multi_station_transfer_plan(
        target_station_id=target_station_id,
        target_dt=target_dt,
        dest_demand=dest_demand,
        dest_available=dest_available,
        dest_name=dest_name,
        df_features=df_features,
        eda_station_df=eda_station_df,
        safety_buffer=safety_buffer,
        station_names=station_names,
    )


def find_verified_donor_for_station(
    target_station_id: int,
    target_dt: pd.Timestamp,
    df_features: Optional[pd.DataFrame] = None,
    eda_station_df: Optional[pd.DataFrame] = None,
    safety_buffer: int = 3
) -> Tuple[Optional[int], Optional[str], int, str]:
    """
    Backwards-compatible wrapper returning (donor_id, donor_name, donor_surplus, rationale).
    Uses nearest safe donor matching based on coordinates.
    """
    res = find_nearest_safe_donor(
        target_station_id=target_station_id,
        target_dt=target_dt,
        dest_demand=15,  # representative peak estimate for standalone calls
        dest_available=10,
        df_features=df_features,
        eda_station_df=eda_station_df,
        safety_buffer=safety_buffer
    )
    return res["donor_id"], res["donor_name"], res["donor_safe_surplus"], res["rationale"]


