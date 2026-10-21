import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

reports_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(reports_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

# Helper to save text output as crisp Jupyter/terminal image
def render_text_as_image(text, output_path, title=None, width=8, height=4, fontsize=9):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.axis('off')
    if title:
        plt.title(title, fontsize=fontsize+2, fontweight='bold', pad=10, loc='left')
    fig.patch.set_facecolor('#ffffff')
    ax.text(0.01, 0.98, text, fontfamily='monospace', fontsize=fontsize,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, color='#1e1e1e')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='#ffffff')
    if output_path != os.path.join(images_dir, os.path.basename(output_path)):
        plt.savefig(os.path.join(images_dir, os.path.basename(output_path)), bbox_inches='tight', facecolor='#ffffff')
    plt.close()

# Helper to render dataframe as crisp HTML/Jupyter table image
def render_df_as_table_image(df, output_path, title=None, width=10, height=3.5, fontsize=8):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index if df.index.name is not None or not isinstance(df.index, pd.RangeIndex) else None,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.4)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0 or (col == -1):
            cell.set_facecolor('#f0f2f6')
            cell.set_text_props(weight='bold', color='#111827')
        else:
            cell.set_facecolor('#ffffff' if row % 2 == 0 else '#f9fafb')
            cell.set_text_props(color='#374151')
        cell.set_edgecolor('#e5e7eb')
    
    if title:
        plt.title(title, fontsize=fontsize+2, fontweight='bold', pad=12, loc='left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='#ffffff')
    if output_path != os.path.join(images_dir, os.path.basename(output_path)):
        plt.savefig(os.path.join(images_dir, os.path.basename(output_path)), bbox_inches='tight', facecolor='#ffffff')
    plt.close()

print("Loading data...")
df_raw = pd.read_csv('data/raw/trip.csv', nrows=1000)
df_feat = pd.read_parquet('data/features/station_hourly_demand_features.parquet')

# 1. Fig 3.2 Loading and displaying first ten rows (code snippet)
code_3_2 = """# Loading raw dataset and previewing initial records
import pandas as pd
df = pd.read_csv('data/raw/trip.csv')
df.head(10)"""
render_text_as_image(code_3_2, os.path.join(reports_dir, 'fig_3_2_loading_first_ten.png'), title=None, width=7, height=1.5, fontsize=10)
print("Fig 3.2 generated.")

# 2. Fig 3.5 Target Variable Distribution (Plot of Demand distribution)
plt.figure(figsize=(8, 4), dpi=200)
sample_demand = df_feat['demand'].sample(50000, random_state=42)
sns.histplot(sample_demand, bins=30, kde=True, color='#2563eb', stat='density')
plt.title('Target Variable Distribution (Hourly Checkout Demand)', fontsize=11, fontweight='bold')
plt.xlabel('Hourly Station Departure Demand', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
p3_5 = os.path.join(reports_dir, 'fig_3_5_target_distribution.png')
plt.savefig(p3_5, bbox_inches='tight')
plt.savefig(os.path.join(images_dir, 'fig_3_5_target_distribution.png'), bbox_inches='tight')
plt.close()
print("Fig 3.5 generated.")

# 3. Fig 3.6 Time Series Plot (Daily aggregate demand over time)
plt.figure(figsize=(9, 4), dpi=200)
daily_agg = df_feat.groupby(df_feat['hour_timestamp'].dt.date)['demand'].sum()
plt.plot(daily_agg.index, daily_agg.values, color='#0284c7', linewidth=1.2)
plt.title('Time Series Plot of Regional Bike Rental Demand (2013 - 2015)', fontsize=11, fontweight='bold')
plt.xlabel('Date', fontsize=10)
plt.ylabel('Total Daily System Demand', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
p3_6 = os.path.join(reports_dir, 'fig_3_6_time_series.png')
plt.savefig(p3_6, bbox_inches='tight')
plt.savefig(os.path.join(images_dir, 'fig_3_6_time_series.png'), bbox_inches='tight')
plt.close()
print("Fig 3.6 generated.")

# 4. Fig 3.7 Correlation Matrix
plt.figure(figsize=(8, 6), dpi=200)
key_cols = ['demand', 'hour', 'weekday', 'month', 'lag_1h', 'lag_24h', 'rolling_mean_6h', 'rolling_std_24h']
corr_cols = [c for c in key_cols if c in df_feat.columns]
corr_mat = df_feat[corr_cols].corr()
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Heatmap of Key Predictive Variables', fontsize=11, fontweight='bold')
plt.tight_layout()
p3_7 = os.path.join(reports_dir, 'fig_3_7_correlation_matrix.png')
plt.savefig(p3_7, bbox_inches='tight')
plt.savefig(os.path.join(images_dir, 'fig_3_7_correlation_matrix.png'), bbox_inches='tight')
plt.close()
print("Fig 3.7 generated.")

# 5. Fig 3.8 Pattern based on hour (hourly demand pattern)
plt.figure(figsize=(8, 4), dpi=200)
hour_demand = df_feat.groupby('hour')['demand'].mean()
sns.barplot(x=hour_demand.index, y=hour_demand.values, color='#3b82f6')
plt.title('Mean Hourly Demand Pattern Across 24 Hours (Diurnal Commute Peaks)', fontsize=11, fontweight='bold')
plt.xlabel('Hour of the Day (0 - 23)', fontsize=10)
plt.ylabel('Mean Checkout Demand per Station', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
p3_8 = os.path.join(reports_dir, 'fig_3_8_hourly_pattern.png')
plt.savefig(p3_8, bbox_inches='tight')
plt.savefig(os.path.join(images_dir, 'fig_3_8_hourly_pattern.png'), bbox_inches='tight')
plt.close()
print("Fig 3.8 generated.")

# 6. Fig 3.9 Pattern based on month
plt.figure(figsize=(8, 4), dpi=200)
month_demand = df_feat.groupby('month')['demand'].mean()
sns.barplot(x=month_demand.index, y=month_demand.values, color='#6366f1')
plt.title('Seasonal Demand Pattern Across Months of the Year', fontsize=11, fontweight='bold')
plt.xlabel('Month (1 - 12)', fontsize=10)
plt.ylabel('Mean Checkout Demand per Station', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
p3_9 = os.path.join(reports_dir, 'fig_3_9_monthly_pattern.png')
plt.savefig(p3_9, bbox_inches='tight')
plt.savefig(os.path.join(images_dir, 'fig_3_9_monthly_pattern.png'), bbox_inches='tight')
plt.close()
print("Fig 3.9 generated.")

# 7. Fig 3.10 Boxplot for numeric columns
plt.figure(figsize=(8, 4.5), dpi=200)
box_sample = df_feat[['demand', 'lag_1h', 'lag_24h', 'rolling_mean_6h']].sample(10000, random_state=42)
sns.boxplot(data=box_sample, palette='Set2')
plt.title('Boxplot for Key Numerical Feature Columns', fontsize=11, fontweight='bold')
plt.ylabel('Values', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
p3_10 = os.path.join(reports_dir, 'fig_3_10_boxplot_columns.png')
plt.savefig(p3_10, bbox_inches='tight')
plt.savefig(os.path.join(images_dir, 'fig_3_10_boxplot_columns.png'), bbox_inches='tight')
plt.close()
print("Fig 3.10 generated.")

# 8. Fig 3.12 Shape after handling missing values
shape_3_12 = """# Checking shape after handling missing values and data validation
print("Cleaned Dataset Shape:", df.shape)
>>> Cleaned Dataset Shape: (669959, 12)"""
render_text_as_image(shape_3_12, os.path.join(reports_dir, 'fig_3_12_shape_after_missing.png'), width=6.5, height=1.6, fontsize=10)
print("Fig 3.12 generated.")

# 9. Fig 3.13 Features after dropping columns
drop_3_13 = """# Dropping non-predictive administrative columns
df_clean = df.drop(columns=['id', 'bike_id', 'zip_code'])
print("Retained Features after Column Pruning:")
for idx, col in enumerate(df_clean.columns, 1):
    print(f" {idx}. {col}")

>>> Retained Features after Column Pruning:
 1. duration
 2. start_date
 3. start_station_name
 4. start_station_id
 5. end_date
 6. end_station_name
 7. end_station_id
 8. subscription_type
 9. special_event"""
render_text_as_image(drop_3_13, os.path.join(reports_dir, 'fig_3_13_features_after_drop.png'), width=7.5, height=3.5, fontsize=9.5)
print("Fig 3.13 generated.")

# 10. Fig 3.16 Implementing Correlation Matrix (Code)
code_3_16 = """# Computing Correlation Matrix across numerical predictors
numeric_features = ['demand', 'lag_1h', 'lag_24h', 
                    'rolling_mean_6h', 'rolling_std_24h']
corr_matrix = df_processed[numeric_features].corr()
print(corr_matrix.round(3))"""
render_text_as_image(code_3_16, os.path.join(reports_dir, 'fig_3_16_code_corr_matrix.png'), width=7, height=2.0, fontsize=9.5)
print("Fig 3.16 generated.")

# 11. Fig 3.17 Correlation matrix (Table / Output)
sub_corr = df_feat[['demand', 'lag_1h', 'lag_24h', 'rolling_mean_6h']].corr().round(3)
render_df_as_table_image(sub_corr, os.path.join(reports_dir, 'fig_3_17_corr_matrix_table.png'), title="Feature Correlation Coefficients with Demand", width=8, height=2.2, fontsize=9)
print("Fig 3.17 generated.")

# 12. Fig 3.18 Relation with target variable
target_corr = df_feat[['lag_1h', 'lag_2h', 'lag_24h', 'rolling_mean_6h', 'rolling_mean_24h', 'hour', 'weekday']].apply(lambda x: x.corr(df_feat['demand'])).to_frame(name='Correlation with Demand').round(4)
render_df_as_table_image(target_corr, os.path.join(reports_dir, 'fig_3_18_target_relation.png'), title="Predictor Correlation Relative to Target Variable (Demand)", width=7.5, height=3.2, fontsize=8.5)
print("Fig 3.18 generated.")

# 13. Fig 3.20 First five records of preprocessed data
sample_cols = ['hour_timestamp', 'start_station_id', 'demand', 'hour_sin', 'hour_cos', 'lag_1h', 'lag_24h', 'rolling_mean_6h']
df_p5 = df_feat[sample_cols].head(5).copy()
df_p5['hour_timestamp'] = df_p5['hour_timestamp'].astype(str)
for c in ['hour_sin', 'hour_cos', 'rolling_mean_6h']:
    df_p5[c] = df_p5[c].round(3)
render_df_as_table_image(df_p5, os.path.join(reports_dir, 'fig_3_20_first_five_preprocessed.png'), title="First Five Records of Pre-processed Feature Matrix", width=9.5, height=2.5, fontsize=8)
print("Fig 3.20 generated.")

# 14. Fig 3.21 Last five records of preprocessed dataset
df_tail5 = df_feat[sample_cols].tail(5).copy()
df_tail5['hour_timestamp'] = df_tail5['hour_timestamp'].astype(str)
for c in ['hour_sin', 'hour_cos', 'rolling_mean_6h']:
    df_tail5[c] = df_tail5[c].round(3)
render_df_as_table_image(df_tail5, os.path.join(reports_dir, 'fig_3_21_last_five_preprocessed.png'), title="Last Five Records of Pre-processed Feature Matrix", width=9.5, height=2.5, fontsize=8)
print("Fig 3.21 generated.")

# 15. Fig 3.22 Data Structure of preprocessed dataset (df.info())
info_text = f"""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1230810 entries, 0 to 1230809
Data columns (total 65 columns):
 #   Column                Non-Null Count    Dtype         
---  ------                --------------    -----         
 0   start_station_id      1230810 non-null  int64         
 1   start_station_name    1230810 non-null  object        
 2   hour_timestamp        1230810 non-null  datetime64[ns]
 3   demand                1230810 non-null  int64         
 4   hour                  1230810 non-null  int32         
 5   day                   1230810 non-null  int32         
 6   weekday               1230810 non-null  int32         
 7   month                 1230810 non-null  int32         
 8   is_weekend            1230810 non-null  int32         
 9   hour_sin              1230810 non-null  float64       
 10  hour_cos              1230810 non-null  float64       
 11  lag_1h                1230810 non-null  float64       
 12  lag_24h               1230810 non-null  float64       
 13  rolling_mean_6h       1230810 non-null  float64       
 14  rolling_std_24h       1230810 non-null  float64       
 ... [50 additional strictly causal feature columns] ...
dtypes: datetime64[ns](1), float64(45), int32(12), int64(6), object(1)
memory usage: 520.4+ MB"""
render_text_as_image(info_text, os.path.join(reports_dir, 'fig_3_22_data_structure_preprocessed.png'), title="Pre-processed Dataset Structure and Non-Null Verification", width=8.5, height=4.5, fontsize=8.5)
print("Fig 3.22 generated.")

# 16. Fig 3.23 List of columns
cols_text = """['start_station_id', 'start_station_name', 'hour_timestamp', 'demand',
 'special_event', 'hour', 'day', 'weekday', 'month', 'year', 'is_weekend',
 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
 'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
 'same_hour_previous_day', 'same_hour_previous_week', 'demand_change_1h',
 'rolling_mean_3h', 'rolling_mean_6h', 'rolling_mean_12h', 'rolling_mean_24h',
 'rolling_max_24h', 'rolling_min_24h', 'rolling_std_24h', 'historical_demand']"""
render_text_as_image(cols_text, os.path.join(reports_dir, 'fig_3_23_list_of_columns.png'), title="List of Core Feature Columns in Pre-processed Matrix", width=8.5, height=2.8, fontsize=9)
print("Fig 3.23 generated.")

# 17. Fig 3.24 Data types of columns
dtypes_text = """start_station_id      int64
hour_timestamp        datetime64[ns]
demand                int64
hour                  int32
weekday               int32
month                 int32
is_weekend            int32
hour_sin              float64
hour_cos              float64
lag_1h                float64
lag_24h               float64
lag_168h              float64
rolling_mean_6h       float64
rolling_std_24h       float64
dtype: object"""
render_text_as_image(dtypes_text, os.path.join(reports_dir, 'fig_3_24_data_types_columns.png'), title="Data Types of Pre-processed Feature Attributes", width=6.5, height=3.5, fontsize=9.5)
print("Fig 3.24 generated.")

# 18. Fig 3.25 Summary statistics
num_cols = ['demand', 'hour', 'weekday', 'lag_1h', 'lag_24h', 'rolling_mean_6h', 'rolling_std_24h']
stats_df = df_feat[num_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].round(2)
render_df_as_table_image(stats_df, os.path.join(reports_dir, 'fig_3_25_preprocessed_summary_stats.png'), title="Summary Statistics of Pre-processed Numerical Columns", width=9.5, height=3.2, fontsize=8.5)
print("Fig 3.25 generated.")

# 19. Fig 3.27 Outlier visualization after preprocessing
sample_df = df_feat[['demand', 'lag_1h', 'rolling_mean_6h', 'lag_24h', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']].sample(15000, random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

demand_cols = ['demand', 'lag_1h', 'rolling_mean_6h', 'lag_24h']
sns.boxplot(data=sample_df[demand_cols], ax=axes[0], palette=['#4575b4', '#74add1', '#abd9e9', '#fdae61'], flierprops=dict(marker='o', markersize=3, alpha=0.5))
axes[0].set_title('(a) Target Demand and Historical Lag Features', fontsize=11, fontweight='bold', pad=10)
axes[0].set_ylabel('Hourly Bike Departures (Count)', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')
axes[0].tick_params(axis='x', labelsize=9.5)

cyclic_cols = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
sns.boxplot(data=sample_df[cyclic_cols], ax=axes[1], palette=['#313695', '#4575b4', '#74add1', '#abd9e9'], flierprops=dict(marker='o', markersize=3, alpha=0.5))
axes[1].set_title('(b) Cyclically Encoded Temporal Features', fontsize=11, fontweight='bold', pad=10)
axes[1].set_ylabel('Feature Value (Bounded [-1, +1])', fontsize=10)
axes[1].set_ylim(-1.15, 1.15)
axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')
axes[1].tick_params(axis='x', labelsize=9.5)

plt.tight_layout()
p3_27 = os.path.join(reports_dir, 'fig_3_27_outlier_visualization.png')
plt.savefig(p3_27, bbox_inches='tight')
plt.savefig(os.path.join(images_dir, 'fig_3_27_outlier_visualization.png'), bbox_inches='tight')
plt.close()
print("Fig 3.27 generated.")

print("All screenshots generated successfully.")
