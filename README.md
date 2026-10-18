# Urban Bike-Sharing Demand Forecasting Using Machine Learning

An end-to-end, production-grade machine learning platform and interactive decision-support system for hourly bicycle departure demand forecasting across 70 transit stations in the San Francisco Bay Area Bike Share network.

---

## 🚲 Project Highlights

- **Predictive Architecture**: Champion **Tuned XGBoost Regressor** trained on **1.23 million continuous hourly observations** generated from 669,959 historical trips.
- **Strict Leakage Prevention**: 54 engineered features adhering strictly to causal time horizons with audited zero temporal data leakage.
- **Decision Support System**: Proactive multi-station bike redistribution and donor rebalancing algorithm with a 3-bike safety buffer rule.
- **High-Performance Analytics**: Columnar **Apache Parquet** caching enabling sub-second multi-resolution query responses and interactive visualizations.
- **Interactive Web Interface**: Production **Streamlit** dashboard featuring live inference, station profile analytics, and multi-resolution demand history exploration.

---

## 📁 Repository Structure

```text
Urban-bike-sharing-demand-prediction/
├── dashboard/                  # Streamlit web application & UI assets
│   ├── app.py                  # Multi-page dashboard with demand prediction & history
│   └── assets/                 # Background and UI branding images
├── data/                       # Datasets & analytical outputs
│   ├── raw/                    # Raw trip transactions (trip.csv)
│   ├── preprocessed/           # Aggregated station hourly demand series
│   ├── features/               # 54-feature matrices (Parquet & CSV) & schemas
│   ├── eda/                    # Exploratory data analysis tables & visual plots
│   └── model_results/          # Evaluation metrics, audits, & residual charts
├── docs/                       # Technical documentation & presentation assets
│   ├── presentation/           # PowerPoint deck, HTML presentation, & generators
│   └── reference_papers/       # Peer-reviewed journal references & project reports
├── models/                     # Serialized production models
│   ├── xgboost_final_model.pkl # Production champion model (Tuned XGBoost)
│   ├── catboost_final_model.pkl# Benchmarked CatBoost model
│   └── random_forest_final_model.pkl # Benchmarked Random Forest model
├── notebooks/                  # Reproducible Jupyter/Colab analysis notebooks
│   ├── eda.ipynb               # Exploratory data analysis
│   ├── preprocessing.ipynb     # Continuous grid construction & temporal alignment
│   ├── feature_engineering.ipynb # 54 causal lag, rolling, and cyclical features
│   ├── final_feature_audit.ipynb # Temporal data leakage audit
│   ├── train_xgboost.ipynb     # XGBoost hyperparameter tuning & evaluation
│   ├── train_catboost.ipynb    # CatBoost training & feature importance
│   └── train_random_forest.ipynb # Random Forest training
├── eda/                        # Modular Python script for exploratory analysis
├── preprocessing/              # Modular Python script for continuous hourly aggregation
├── feature_engineering/        # Modular Python script for causal feature engineering
├── model_training/             # Training pipelines for ensemble models
├── model_evaluation/           # Leakage audits, residual analyses, & metric benchmarks
├── redistribution/             # Rebalancing engine & multi-station donor heuristics
├── tests/                      # Automated test suite (9 test suites, 100% pass)
├── requirements.txt            # Production environment dependencies
├── main.py                     # Report generation pipeline script
└── .gitignore                  # Git exclusions for caches, environments, & large files
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 to 3.14
- Virtual environment (recommended)

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/IsabellaDennis/Urban-bike-sharing-demand-prediction.git
cd Urban-bike-sharing-demand-prediction
pip install -r requirements.txt
```

### 3. Launching the Interactive Dashboard

Launch the Streamlit web application:

```bash
streamlit run dashboard/app.py
```

The application will be available at `http://localhost:8501`.

---

## 🧪 Testing and Verification

Run the comprehensive test suite to verify prediction accuracy, date compatibility, UI components, and redistribution logic:

```bash
# Run via pytest
pytest tests/

# Run complete integration test suite
python tests/run_all_event_tests.py
```

All 9 test suites validate:
1. Station-specific and hour-specific machine learning inference.
2. Defensible special-event adjustments and date compatibility.
3. Multi-station donor rebalancing and distance-based ordering.
4. Demand history temporal filters, peak-hour calculations, and state persistence.
5. Streamlit AppTest headless UI navigation and metric rendering.

---

## 📊 Model Performance Comparison

| Model | Test $R^2$ | Test RMSE | Test MAE | Training Runtime |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.7431** | **1.713** | **0.869** | **Fastest** |
| Random Forest | 0.7423 | 1.716 | 0.863 | High Memory |
| CatBoost | 0.7329 | 1.747 | 0.892 | Moderate |

---

## ⚖️ License & Attribution

Developed for academic and mobility research using the San Francisco Bay Area Bike Share open data.
