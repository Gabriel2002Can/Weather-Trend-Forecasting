# Weather Trend Forecasting
A comprehensive data science project for analyzing global weather data, performing exploratory data analysis (EDA), outlier detection, and building forecasting and classification models for weather trends and conditions.

# Project Overview
This project explores a large global weather dataset (GlobalWeatherRepository.csv) containing current and historical weather observations from numerous locations worldwide. The notebook covers the full data science pipeline: data loading and cleaning, exploratory analysis, anomaly detection, visualization, and machine learning modeling (including time-series forecasting and classification of weather conditions).

Key Objectives:
> - Understand global weather patterns and distributions.
> - Clean and preprocess the data for modeling.
> - Detect anomalies/outliers in weather observations.
> - Build models for forecasting temperature trends and classifying weather conditions.
> - Generate insights and visualizations.

# Dataset

- **File:** GlobalWeatherRepository.csv
- **Size:** ~140,923 rows × 41 columns
- **Features include:** country, location_name, latitude/longitude, temperature (C/F), condition_text, wind metrics, precipitation, humidity, cloud cover, air quality indices (PM2.5, PM10, etc.), sunrise/sunset, moon phase, and more.
- **Time coverage:** Spans multiple dates (2024–2026 in the sample).

# Methodology
1. **Data Import & Cleaning**

    - Loaded data with pandas.
    - Handled missing values (none found in key checks).
    - Standardized the target weather condition column (y / condition_text): stripped whitespace, lowercased, grouped rare classes into "rare" or "rest" categories to handle imbalance.
    - Feature engineering and type conversions for modeling.

2. **Exploratory Data Analysis (EDA)**

    - Weather condition distribution (bar plots of top conditions like "partly cloudy", "sunny", etc.).
    - Visualizations of temperature, wind, precipitation, air quality, and their relationships.
    - Geographic and temporal insights.
    - Correlation analysis and summary statistics.

3. **Outlier/Anomaly Detection**

    - Used Isolation Forest (contamination=0.02) to identify anomalous weather observations.
    - Analyzed outlier distribution by weather type and visualized proportions.

4. **Modeling**
   
    **Time-Series Forecasting:**

     - Prophet model for temperature forecasting (trend, seasonality).
     - XGBoost Regressor for temperature forecasting.
     - Ensemble of both models for improved results.
     - Good performance achieved.

    **Classification Models:**
  
     - Random Forest Classifier, XGBoost Classifier, and Voting Classifier for predicting weather conditions (condition_text / y).
     - Handled multi-class classification with rare class grouping.
     - Evaluated with accuracy, classification reports, confusion matrices, and cross-validation.
     - High performance achieved (~86%).

5. **Model Evaluation**

    - **Metrics:** Accuracy, Precision, Recall, F1, MSE/MAE for regression.
    - Cross-validation results.
    - Feature importance (from tree-based models).
    - Visualizations of predictions vs. actuals.

# **Key Insights**

- Dominant weather conditions globally: partly cloudy and sunny.
- Strong correlations between temperature, feels-like temperature, humidity, and cloud cover.
- Air quality variations by location and conditions.
- Outliers often linked to extreme weather events (heavy rain, snow, storms).
- Models effectively capture patterns for forecasting and classification, with room for hyperparameter tuning and additional features (e.g., more temporal lags).

# **Repository Structure**
```
Weather-Trend-Forecasting/
├── GlobalWeatherRepository.csv     # Raw dataset
├── weather_trend.ipynb             # Main Jupyter Notebook (all code + outputs)
├── README.md                       # This file
└── .ipynb_checkpoints/             # (Auto-generated)
```
# **How to Run**

1. Clone the repo:
```
git clone https://github.com/Gabriel2002Can/Weather-Trend-Forecasting.git
cd Weather-Trend-Forecasting
```
2. Install dependencies (recommended in a virtual environment):
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet
```
3. Open the notebook:
```
jupyter notebook weather_trend.ipynb
```

# **Technologies Used**

- **Python:** pandas, NumPy, Matplotlib, Seaborn
- **ML:** scikit-learn, XGBoost, Prophet, Isolation Forest
- **Visualization:** Matplotlib/Seaborn
- **Environment:** Jupyter Notebook

# **License**
Open for learning and reference. Feel free to fork and extend!
