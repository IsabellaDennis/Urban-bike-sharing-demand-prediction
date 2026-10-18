"""
Unit and Integration Test Suite for Multi-Station Bike Redistribution Logic.
Tests all requirements:
1. Single donor covering the shortage.
2. Two or more donors required to cover the shortage.
3. Multiple donors whose combined surplus is insufficient.
4. Donors with zero safe surplus excluded.
5. Destination station excluded from donor candidates.
6. Donor ordering strictly by verified geographic distance.
7. Transfer quantities never exceeding safe surplus or remaining shortage.
8. Consistency between displayed shortage and transfer allocations.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from redistribution.redistribution import (
    find_multi_station_transfer_plan,
    find_nearest_safe_donor,
    compute_network_rebalancing,
    get_station_coords,
    haversine_distance_km,
    get_station_capacity,
    estimate_available_bikes,
    calculate_station_shortage_surplus,
)

print("=" * 80)
print("RUNNING MULTI-STATION BIKE REDISTRIBUTION VERIFICATION TEST SUITE")
print("=" * 80)

passed_count = 0
total_tests = 8

def assert_test(condition, test_num, name, details):
    global passed_count
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] Test {test_num}: {name}")
    print(f"       Details: {details}\n")
    assert condition, f"Test {test_num} failed: {details}"
    passed_count += 1

target_dt = pd.Timestamp("2015-06-15 08:00:00")

# -------------------------------------------------------------------------
# Test 1: Single donor covering the entire shortage
# -------------------------------------------------------------------------
# Destination needs 5 bikes. Donor 69 has 15 surplus bikes.
res1 = find_multi_station_transfer_plan(
    target_station_id=70,
    target_dt=target_dt,
    dest_demand=25,
    dest_available=20,  # shortage = 5
    dest_name="San Francisco Caltrain",
    safety_buffer=3
)
t1_pass = (
    res1["destination_shortage"] == 5
    and res1["is_multi_donor"] is False
    and len(res1["donors"]) == 1
    and res1["donors"][0]["recommended_transfer"] == 5
    and res1["total_transferred"] == 5
    and res1["unmet_demand"] == 0
    and res1["status"] == "safe_donor_found"
    and res1["is_partial"] is False
)
assert_test(
    t1_pass, 1,
    "Single donor covering the shortage",
    f"Shortage: 5 -> Transferred: {res1['total_transferred']}, Unmet: {res1['unmet_demand']}, Donors: {len(res1['donors'])} ({res1['donors'][0]['donor_name']})"
)

# -------------------------------------------------------------------------
# Test 2: Two or more donors required to cover the shortage
# -------------------------------------------------------------------------
# Build a controlled snapshot dataframe:
# Destination: Station 70 (Townsend at 4th)
# Donor A: Station 69 (0.02 km away) -> capacity 20, demand 5, buffer 3 -> surplus = 20 - 5 - 3 = 12
# Donor B: Station 61 (0.62 km away) -> capacity 20, demand 5, buffer 3 -> surplus = 12
# Destination needs 20 bikes (demand 30, available 10).
# Station 69 should give 12 bikes (Step 1). Station 61 should give remaining 8 bikes (Step 2).
controlled_df = pd.DataFrame([
    {"hour_timestamp": target_dt, "start_station_id": 70, "start_station_name": "Station 70", "demand": 30},
    {"hour_timestamp": target_dt, "start_station_id": 69, "start_station_name": "Station 69", "demand": 5},
    {"hour_timestamp": target_dt, "start_station_id": 61, "start_station_name": "Station 61", "demand": 5},
])
res2 = find_multi_station_transfer_plan(
    target_station_id=70,
    target_dt=target_dt,
    dest_demand=30,
    dest_available=10,  # shortage = 20
    dest_name="Station 70",
    df_features=controlled_df,
    safety_buffer=3
)
t2_pass = (
    res2["destination_shortage"] == 20
    and res2["is_multi_donor"] is True
    and len(res2["donors"]) == 2
    and res2["donors"][0]["donor_id"] == 69
    and res2["donors"][0]["recommended_transfer"] == res2["donors"][0]["safe_surplus"]  # all surplus taken
    and res2["donors"][1]["donor_id"] == 61
    and res2["total_transferred"] == 20
    and res2["unmet_demand"] == 0
    and res2["status"] == "safe_donor_found"
    and res2["is_partial"] is False
)
assert_test(
    t2_pass, 2,
    "Two or more donors required to cover the shortage",
    f"Shortage: 20 -> Step 1: {res2['donors'][0]['recommended_transfer']} from Station {res2['donors'][0]['donor_id']} ({res2['donors'][0]['distance_km']} km) | Step 2: {res2['donors'][1]['recommended_transfer']} from Station {res2['donors'][1]['donor_id']} ({res2['donors'][1]['distance_km']} km) | Total: {res2['total_transferred']}, Unmet: {res2['unmet_demand']}"
)

# -------------------------------------------------------------------------
# Test 3: Multiple donors whose combined surplus is insufficient (partial fulfillment)
# -------------------------------------------------------------------------
# Destination needs 40 bikes.
# Donor 69 has surplus 12. Donor 61 has surplus 12. Total surplus = 24.
# Total transferred = 24. Unmet demand = 16. Status = partial_donor_found.
res3 = find_multi_station_transfer_plan(
    target_station_id=70,
    target_dt=target_dt,
    dest_demand=50,
    dest_available=10,  # shortage = 40
    dest_name="Station 70",
    df_features=controlled_df,
    safety_buffer=3
)
t3_pass = (
    res3["destination_shortage"] == 40
    and res3["is_multi_donor"] is True
    and len(res3["donors"]) == 2
    and res3["total_transferred"] == (res3["donors"][0]["safe_surplus"] + res3["donors"][1]["safe_surplus"])
    and res3["unmet_demand"] == (40 - res3["total_transferred"])
    and res3["unmet_demand"] > 0
    and res3["status"] == "partial_donor_found"
    and res3["is_partial"] is True
)
assert_test(
    t3_pass, 3,
    "Multiple donors whose combined surplus is insufficient",
    f"Shortage: 40 -> Transferred: {res3['total_transferred']} (across {len(res3['donors'])} donors) | Unmet: {res3['unmet_demand']} | Status: {res3['status']}"
)

# -------------------------------------------------------------------------
# Test 4: Donors with zero or negative safe surplus are excluded
# -------------------------------------------------------------------------
# Station 65 has high demand (demand=18) -> available=16 -> rem = -2 -> safe_surplus = 0.
# It MUST NOT be included as a donor.
zero_surplus_df = pd.DataFrame([
    {"hour_timestamp": target_dt, "start_station_id": 70, "start_station_name": "Station 70", "demand": 25},
    {"hour_timestamp": target_dt, "start_station_id": 65, "start_station_name": "Station 65 (Zero Surplus)", "demand": 18},
    {"hour_timestamp": target_dt, "start_station_id": 61, "start_station_name": "Station 61 (Positive Surplus)", "demand": 5},
])
res4 = find_multi_station_transfer_plan(
    target_station_id=70,
    target_dt=target_dt,
    dest_demand=25,
    dest_available=10,  # shortage = 15
    dest_name="Station 70",
    df_features=zero_surplus_df,
    safety_buffer=3
)
donor_ids_res4 = [d["donor_id"] for d in res4["donors"]]
t4_pass = (
    65 not in donor_ids_res4
    and 61 in donor_ids_res4
    and all(d["safe_surplus"] > 0 for d in res4["donors"])
)
assert_test(
    t4_pass, 4,
    "Donors with zero safe surplus are excluded",
    f"Candidate with 0 surplus (Station 65) excluded. Donors selected: {donor_ids_res4}"
)

# -------------------------------------------------------------------------
# Test 5: Destination excluded from donor candidates
# -------------------------------------------------------------------------
# Destination station 70 must never appear in its own donor list
res5 = find_multi_station_transfer_plan(
    target_station_id=70,
    target_dt=target_dt,
    dest_demand=25,
    dest_available=10,
    dest_name="Station 70",
    df_features=controlled_df,
    safety_buffer=3
)
t5_pass = (
    70 not in [d["donor_id"] for d in res5["donors"]]
    and res5["destination_id"] == 70
)
assert_test(
    t5_pass, 5,
    "Destination station excluded from donor candidates",
    f"Destination ID: {res5['destination_id']} | Donor IDs: {[d['donor_id'] for d in res5['donors']]}"
)

# -------------------------------------------------------------------------
# Test 6: Donor ordering by verified geographic distance
# -------------------------------------------------------------------------
# Three stations with positive surplus:
# Station 69 (0.02 km away)
# Station 61 (0.62 km away)
# Station 50 (2.09 km away)
# Destination needs 30 bikes, so all 3 will be used.
# They MUST appear in strict ascending distance order: 69, 61, 50.
three_donors_df = pd.DataFrame([
    {"hour_timestamp": target_dt, "start_station_id": 70, "start_station_name": "Station 70", "demand": 40},
    {"hour_timestamp": target_dt, "start_station_id": 50, "start_station_name": "Station 50 (Ferry Building)", "demand": 5},
    {"hour_timestamp": target_dt, "start_station_id": 61, "start_station_name": "Station 61 (Townsend)", "demand": 5},
    {"hour_timestamp": target_dt, "start_station_id": 69, "start_station_name": "Station 69 (Townsend 2)", "demand": 5},
])
res6 = find_multi_station_transfer_plan(
    target_station_id=70,
    target_dt=target_dt,
    dest_demand=40,
    dest_available=10,  # shortage = 30
    dest_name="Station 70",
    df_features=three_donors_df,
    safety_buffer=3
)
distances = [d["distance_km"] for d in res6["donors"]]
order_ids = [d["donor_id"] for d in res6["donors"]]
t6_pass = (
    distances == sorted(distances)
    and order_ids[0] == 69  # closest (0.02 km)
    and order_ids[1] == 61  # second (0.62 km)
    and order_ids[2] == 50  # third (2.09 km)
)
assert_test(
    t6_pass, 6,
    "Donor ordering by distance",
    f"Ordered donors: {order_ids} with distances: {distances} km"
)

# -------------------------------------------------------------------------
# Test 7: Transfer quantities never exceed safe surplus or remaining shortage
# -------------------------------------------------------------------------
# Every donor's recommended_transfer must satisfy:
# recommended_transfer <= safe_surplus
# sum(recommended_transfer) <= destination_shortage
t7_pass = True
for r in [res1, res2, res3, res4, res5, res6]:
    for d in r["donors"]:
        if d["recommended_transfer"] > d["safe_surplus"]:
            t7_pass = False
        if d["recommended_transfer"] <= 0:
            t7_pass = False
    if r["total_transferred"] > r["destination_shortage"]:
        t7_pass = False
assert_test(
    t7_pass, 7,
    "Transfer quantities never exceed safe surplus or remaining shortage",
    "Verified across all test cases: for every donor, 0 < transfer <= safe_surplus and sum(transfers) <= shortage."
)

# -------------------------------------------------------------------------
# Test 8: Consistency between displayed shortage and transfer allocations
# -------------------------------------------------------------------------
# destination_shortage MUST equal total_transferred + unmet_demand
t8_pass = True
for r in [res1, res2, res3, res4, res5, res6]:
    if r["destination_shortage"] != (r["total_transferred"] + r["unmet_demand"]):
        t8_pass = False
assert_test(
    t8_pass, 8,
    "Consistency between displayed shortage and transfer allocations",
    "Verified invariant shortage == total_transferred + unmet_demand holds identically across all cases."
)

print("=" * 80)
print(f"ALL {passed_count}/{total_tests} MULTI-STATION REDISTRIBUTION TESTS PASSED SUCCESSFULLY!")
print("=" * 80)
