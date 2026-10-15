import os
import re

# Comprehensive speaker notes for all 14 slides
DETAILED_NOTES = {
    1: """Good morning respected guide Mr. Sibu Skaria, external examiners, and panel members.
My name is Isabella Dennis, Roll Number 30, from the Department of Computer Applications.
Today, I am presenting my MCA Mini Project on 'Urban Bike Sharing Demand Forecasting: Predicting Station-Level Hourly Bike Demand Using Machine Learning'.
In this project, we address a critical logistical problem in modern smart cities: forecasting hourly bicycle departures at each docking station across a regional network using historical transaction data.
Notice that our target is explicitly formulated as station-level hourly departure demand, which provides a direct and actionable metric for transit operators to maintain system balance.""",

    2: """Slide 2 establishes our Problem Statement, Project Objectives, and Literature Review foundation.

• Card 1 - Problem Statement:
In urban bike-sharing systems, commuters travel one-way during morning and evening rush hours. This causes severe spatial imbalance: residential stations run out of bikes in the morning, while commercial and transit hubs overflow. Fleet managers cannot reposition bikes effectively without knowing upcoming hourly departure demand in advance. Furthermore, raw logs only record active trips—missing quiet hours completely—which causes naive models to severely over-predict nighttime demand. Hence, our target is defined as hourly trip departures starting from each station.

• Card 2 - Project Objectives:
First, we formulate a continuous citywide grid of 1,230,810 station-hour records across 70 stations and 17,583 hours with explicit zero-demand handling. Second, we engineer 54 strictly causal features across calendar, cyclical time, interaction terms, historical lags, and rolling averages with zero data leakage. Third, we benchmark Random Forest, XGBoost, and CatBoost under strict chronological time splits. Fourth, we identify key demand drivers to assist transit managers in proactive bike redistribution.

• Card 3 - Supporting Literature & Gap:
Aydın et al. (2023) demonstrated that gradient tree boosting algorithms like CatBoost and XGBoost excel on non-linear hourly demand patterns. Karunanithi et al. (2024) proved that tree ensembles significantly outperform neural networks on structured tabular time features. Yao et al. (2024) modeled spatial network dynamics on this SF Bay Area dataset. Our work addresses the critical research gap of full zero-demand grid expansion combined with a strict chronological leakage audit.""",

    3: """Slide 3 introduces Stage 1 of our pipeline: Raw Data Collection.

• Dataset Overview: We use the real-world San Francisco Bay Area Bike Share dataset from Kaggle, specifically trip.csv.
• Key Specifications: It records 669,959 raw trip transactions spanning August 29, 2013 to August 31, 2015. That represents 733 days, or two full years, covering 17,583 continuous hourly timestamps across 70 unique docking stations.
• Why Hourly Departures as Target?: Trip durations in raw data have extreme outliers ranging from 60 seconds to over 200 days due to unreturned bikes. Meanwhile, trip.csv only logs departure events, not live dock occupancy. Formulating the target as hourly departures provides a clean, stable, and operationally actionable metric.
• Visuals: Figure 3.1 on the right shows the dataset shape (669,959 rows by 12 columns) and data types. Figure 3.3 displays the first 10 raw ride receipts.
• Stage 1 Summary: We begin with 669,959 raw checkout records and 12 columns.""",

    4: """Slide 4 presents our Exploratory Data Analysis, examining data quality and rider behavior patterns.

• Data Quality Audit (Top Card):
The raw data is remarkably high quality. Only 6,619 records had missing values in zip_code (~0.98%), which is an optional personal field. All other 11 columns have zero missing values. Duplicate check revealed exactly zero duplicates across all 669k records. User breakdown shows that 84% of riders are Annual Subscribers (daily office commuters) and 16% are Casual Customers (tourists).

• Rider Behavior Patterns (Bottom Card):
On weekdays, demand displays a sharp twin commuter rush hour peak at 8:00 AM and 5:00 PM. On weekends, the commuter spikes vanish completely, replaced by a gentle recreational midday curve between 12:00 PM and 3:00 PM. Station demand varies drastically: SF Caltrain recorded over 49,000 trips, while quiet suburban stations had fewer than 1,000 trips over two years.
• Visuals: The top graph shows the twin-peak weekday commute curve vs the weekend curve. The bottom graph confirms higher total demand on workdays.
• Key Insight: Demand fluctuates heavily by hour, day of week, and station, proving the necessity of automated machine learning forecasts.""",

    5: """Slide 5 describes Stage 2: Data Preprocessing and Solving the Zero-Demand Problem.

• Step 1 - Hourly Rounding: Exact ride timestamps contain arbitrary minutes and seconds. We floor every start datetime down to the beginning of the hour (e.g., 2013-08-29 09:08:24 becomes 2013-08-29 09:00:00). This groups all rides within each 60-minute window.
• Step 2 - The Zero-Demand Problem: In raw logs, if nobody rents a bike at Station 12 at 3:00 AM, no row is recorded. If we only train on rows where rides occurred, the model never learns when demand is zero and will severely over-predict nighttime demand.
• Step 3 - Grid Expansion: To fix this, we generate a complete Cartesian product of 70 stations multiplied by 17,583 hours = exactly 1,230,810 station-hour records. For active hours, we record actual trip counts; for quiet hours, we explicitly assign demand = 0.
• Step 4 - Event Calendar: We extracted city holidays and festivals from trip.csv into a lookup table so that every newly created station-hour row inherits its holiday status (e.g., Thanksgiving, Christmas).
• Stage 2 Summary: Row count expanded from 669,959 trips into 1,230,810 station-hour records across 3 base features: station_id, hour_timestamp, and demand.""",

    6: """Slide 6 details Stage 3: Raw Dataset Column Selection, auditing why each of the 12 raw columns was kept, transformed, or removed.

• id: Removed, as it is simply an individual ride transaction receipt with no predictive power.
• duration: Removed, because it is future leakage—we only know duration after a trip finishes, which is unknown when predicting upcoming departures!
• start_date: Transformed, by flooring it to the hour to group rides into hourly departure buckets.
• start_station_name & start_station_id: Kept, as station identifiers.
• end_date, end_station_name, end_station_id: Removed, because destination is unknown when predicting departures at the start of an hour. Including them would introduce future target leakage.
• bike_id: Removed, because individual bike serial numbers do not affect aggregate station-level demand.
• subscription_type & zip_code: Removed, as they are individual rider attributes rather than station totals, and zip code has missing values.
• special_event: Kept, to identify holidays.
• Stage 3 Summary: Across all 1,230,810 rows, the 12 raw columns are filtered down to core station/time base identifiers.""",

    7: """Slide 7 covers Stage 4: Feature Engineering, where we extract 62 candidate features across 1,230,810 rows.
Consider an illustrative scenario: predicting demand for Tuesday, Oct 15, 2013 at 8:00 AM at Station #2 (San Jose Train Station).

• Card 1 - Calendar & Flags (14 features): Captures hour (8), weekday (Tuesday), is_weekend (0), and morning_peak (1). This informs the model that a busy Tuesday morning rush is occurring.
• Card 2 - Cyclical Sine/Cosine (8 features): Solves the clock gap problem. In linear numbers, 11 PM (23) and 12 AM (0) are 23 units apart, but in reality they are 1 hour apart! Sine and cosine place 24 hours on a continuous 360-degree circle so time flows seamlessly without artificial boundary jumps.
• Card 3 - Temporal Interactions (4 features): weekday_hour (1 × 8 = 8) gives heavy weight to workday rush hours, while weekend_hour (0 × 8 = 0) isolates Sunday morning traffic.
• Card 4 - Past Demand Lags (13 features): lag_1h captures immediate 1-hour momentum (12 trips at 7 AM). lag_24h captures demand yesterday at 8 AM (14 trips). lag_168h captures demand last Tuesday at 8 AM (15 trips). Past demand provides strong baseline anchors.
• Card 5 - Rolling Window Statistics (15 features): rolling_mean_3h (5.7 trips/hr) filters single-hour spikes. rolling_mean_24h (2.1 trips/hr) provides a daily baseline. lag_1h_ratio_24h (5.71) indicates the station is running much hotter than normal.
• Stage 4 Summary: 1,230,810 rows by 62 candidate features generated.""",

    8: """Slide 8 presents Stage 5: our Feature Leakage Audit.
Data leakage happens when a feature accidentally uses information that would not be available in real life at prediction time. If uncorrected, the model achieves artificial 99% accuracy in training but fails completely in deployment.

We audited all 62 features and removed 8 non-causal features:
• 1. Removed 3 Demand Deltas: Features like demand_change_1h used the formula (current_demand - lag_1h). But when predicting at 8:00 AM, current demand is the target answer we are trying to predict! Using it was cheating.
• 2. Removed 5 Global Averages: Features like historical_mean averaged demand over the entire two-year dataset (2013 to 2015). That meant 2013 records were using future numbers from 2015!
• Strict Lags Verified: We confirmed that lag_1h, lag_24h, and lag_168h look strictly backwards in time.
• Stage 5 Summary: 62 candidate features minus 8 removed equals 54 final features. All 54 features strictly use information available prior to the prediction timestamp.""",

    9: """Slide 9 explains Stage 6: our Chronological Data Partitioning strategy.

• Why No Random Shuffling?: In standard tabular machine learning, random train_test_split is common. In time-series forecasting, this is strictly forbidden! Random shuffling allows the model to use Wednesday to predict Tuesday, or June to predict May. It memorizes future answers to predict the past.

• Strict Chronological Splits:
1. Training Set (70%): 861,560 rows from Aug 29, 2013 to Jan 24, 2015. The model learns baseline traffic, rush hour dynamics, and station profiles across 17 full months.
2. Validation Set (15%): 184,590 rows from Jan 24, 2015 to May 14, 2015. Used exclusively to tune hyperparameters like tree depth and learning rate without ever touching the test set.
3. Test Set (15%): 184,660 rows from May 14, 2015 to Aug 31, 2015. Kept strictly locked until final evaluation to test true real-world generalization on unseen future months.
• Stage 6 Summary: 54 causal features evaluated across Train (861k), Validation (184k), and Test (184k) rows.""",

    10: """Slide 10 details our Model Selection: how each of the three tree-based algorithms works.

• Model 1 - Random Forest (Bagging):
Random Forest builds hundreds of decision trees simultaneously in parallel. Each tree votes independently, and we take their average. For example, if Tree 1 predicts 14 trips, Tree 2 predicts 16, and Tree 3 predicts 15, the average prediction is 15 trips. Averaging independent trees reduces variance and handles noisy data well.

• Model 2 - XGBoost (Gradient Boosting):
XGBoost builds decision trees sequentially in a line, where each new tree directly predicts and corrects the previous tree's residual error. For example, if actual demand is 15 and Tree 1 predicts 11, the residual error is +4. Tree 2 is trained specifically to predict that +4 mistake! Repeating this over 600 trees makes it exceptionally powerful at learning sharp commuter rush spikes.

• Model 3 - CatBoost (Ordered Boosting):
CatBoost is also a sequential boosting algorithm. It uses ordered boosting to prevent target leakage during tree building. It provides an independent boosting benchmark and offers extremely fast prediction speeds during real-world inference.""",

    11: """Slide 11 presents our final Model Results evaluated on 184,660 completely unseen test rows.

• Performance Table:
- Random Forest: Test MAE = 0.439206 | RMSE = 0.908465 | R² = 0.731390
- XGBoost: Test MAE = 0.430550 | RMSE = 0.898213 | R² = 0.737418
- CatBoost: Test MAE = 0.436126 | RMSE = 0.897665 | R² = 0.737738

• Detailed Analysis by Card:
1. Random Forest: Delivers solid baseline performance, explaining about 73.1% of demand variation. It confirms that bagging decision trees creates a reliable, stable baseline on tabular time features.
2. XGBoost: Achieved the lowest MAE (0.430550) among all three models. This means it has the lowest average absolute error on hourly trip predictions.
3. CatBoost: Achieved the lowest RMSE (0.897665) and highest R² (0.737738). Lower RMSE means better performance when penalizing large surge mistakes.
• Metric Interpretation: MAE of ~0.43 means our forecast is off by less than half a bike per station-hour on average. R² of ~0.74 means the models explain nearly 74% of the total variation in demand on unseen future data.""",

    12: """Slide 12 examines Feature Importance: what influenced the predictions across all models.

• Top 5 Features Across All Models:
1. same_hour_previous_week: Demand at this exact station 7 days ago at the same hour was the single most powerful predictor.
2. lag_168h: Demand 168 hours ago, reinforcing identical weekly commuter travel habits.
3. peak_hour: Morning rush (7 to 9 AM) and evening rush (4 to 7 PM) binary flags.
4. lag_1h: Departures in the immediately preceding hour, capturing real-time station momentum.
5. lag_24h / same_hour_prev_day: Demand yesterday at the same hour.

• Visual Insights:
The top chart confirms that all three algorithms place the heaviest weight on 7-day past demand and rush hour flags.
The bottom chart displays actual vs predicted departures for CatBoost on the test set. Notice that prediction errors are tightly and symmetrically clustered around zero, confirming consistent calibration without systematic bias.""",

    13: """Slide 13 outlines our roadmap for the Next Phase of implementation:

1. Build Streamlit UI: Develop an interactive, user-friendly web interface using Python Streamlit.
2. Connect Trained Model: Load the serialized CatBoost/XGBoost model into the backend pipeline for instant inference.
3. Station and Time Dropdowns: Allow transit managers to select any of the 70 stations, a target date, and an hour from simple dropdowns.
4. Display Predicted Demand: Instantly display the forecasted number of bike checkouts for that station-hour.
5. Interactive Visualizations: Show 24-hour historical curves and expected peak rush hour demand.
6. Redistribution Support: Provide proactive rebalancing recommendations so transit operators can move bikes before shortages occur.
7. End-to-End Testing: Perform comprehensive testing across edge cases, holidays, and high-traffic stations to ensure production reliability.""",

    14: """Slide 14 concludes our presentation.

• Summary of Achievements:
We successfully resolved the zero-demand problem by expanding 669k trips into a continuous 1.23 million row grid, engineered 54 strictly causal features with zero leakage, and proved that tree ensemble models explain approximately 74% of demand variation on unseen future test data.

I would like to express my sincere gratitude to my project guide, Mr. Sibu Skaria, for his continuous guidance, and to the Department of Computer Applications.
Thank you, respected panel members. I am now ready to take your questions and discussion."""
}

def update_presentation_py():
    with open("generate_presentation.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Replace each set_notes(sX, ...) with the detailed notes using triple quotes
    for i in range(1, 15):
        pattern = rf'set_notes\(s{i},[\s\S]*?\)\n'
        # Clean triple quotes if any inside notes
        clean_note = DETAILED_NOTES[i].replace('"""', "'''")
        replacement = f'set_notes(s{i}, """{clean_note}""")\n'
        content, count = re.subn(pattern, replacement, content, count=1)
        print(f"Updated set_notes for slide {i}: {count} replacements")

    with open("generate_presentation.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("generate_presentation.py updated with detailed notes.")

def update_build_html_presentation_py():
    with open("build_html_presentation.py", "r", encoding="utf-8") as f:
        content = f.read()

    for i in range(1, 15):
        # Match data-notes="..." on slide i
        pattern = rf'(<section class="slide[^"]*" id="slide-{i}" data-notes=")([^"]*)(")'
        escaped_notes = DETAILED_NOTES[i].replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', ' ')
        replacement = rf'\g<1>{escaped_notes}\g<3>'
        content, count = re.subn(pattern, replacement, content)
        print(f"Updated HTML data-notes for slide {i}: {count} replacements")

    with open("build_html_presentation.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("build_html_presentation.py updated with detailed notes.")

if __name__ == "__main__":
    update_presentation_py()
    update_build_html_presentation_py()
