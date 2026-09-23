# 🚦 TrafficPulse AI — Traffic, Congestion & Accident Analytics

> **A complete, professional Data Analysis + AI dashboard** built with Python and Streamlit for analyzing urban traffic patterns, congestion levels, and accident occurrences.

---

## 📌 Project Description

TrafficPulse AI is an end-to-end data analytics and machine learning project that ingests real or synthetic urban traffic data and delivers interactive, insight-rich dashboards. It covers the full data science pipeline — from raw data ingestion and cleaning to exploratory data analysis, statistical summaries, interactive Plotly visualizations, geographic hotspot mapping, and scikit-learn ML models — all within a single Streamlit application.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 📂 Data Loading | Upload any traffic CSV **or** auto-generate realistic synthetic data |
| 🧹 Data Cleaning | Duplicate removal, missing-value imputation, outlier clipping |
| 🔧 Feature Engineering | Peak-hour flag, time-period label, accident rate, vehicle mix ratios |
| 📊 EDA & Statistics | Distribution plots, temporal trends, correlation heatmap |
| 🏠 KPI Dashboard | 7 live KPI cards: records, volume, accidents, rate, speed, congestion, high-risk |
| 📈 Traffic Analysis | Hourly/daily/monthly trends, vehicle-type pie, road-type comparison |
| ⚠️ Accident Analysis | By weather, road condition, location, severity, hour, day |
| 🚧 Congestion Analysis | Density distributions, speed-volume scatter, weather stacked bars |
| 🗺️ Map View | Scatter & density heatmap (Mapbox, requires coordinates) |
| 🤖 ML Predictions | Congestion & accident prediction with 3 model options |
| 📥 CSV Download | Download filtered dataset and ML classification report |
| 💡 Auto Insights | Plain-language automatic observations from loaded data |

---

## 🎯 Objectives

1. Discover temporal and spatial patterns in urban traffic flow.
2. Identify weather and road conditions that correlate with higher accident risk.
3. Quantify congestion levels and understand their drivers.
4. Demonstrate end-to-end ML pipeline for traffic prediction tasks.
5. Provide a shareable, interactive dashboard suitable for stakeholder presentations.

---

## 🗄️ Dataset Description

The project is compatible with the **TrafficPulse AI Dataset** (real or synthetic).

| Column | Description |
|---|---|
| `Record_ID` / `id` | Unique record identifier |
| `Date` | Observation date |
| `Time` | Observation time (HH:MM:SS) |
| `Day_of_Week` | Day name (Monday … Sunday) |
| `Month` | Month number (1–12) |
| `Year` | Calendar year |
| `hour` | Hour of day (0–23) |
| `Location` | Named location / intersection |
| `Latitude`, `Longitude` | GPS coordinates (if available) |
| `Road_Type` | Arterial / Highway / Urban / Residential / Expressway |
| `Road_Condition` | Good / Fair / Poor |
| `Weather` | Clear / Rain / Fog / Cloudy / Storm |
| `Temperature_C` | Ambient temperature in °C |
| `Visibility_km` | Visibility distance in km |
| `Traffic_Volume` | Total volume count |
| `Average_Speed_kmph` | Mean vehicle speed (km/h) |
| `Vehicle_Count` | Total vehicles observed |
| `Car_Count`, `Bike_Count`, `Bus_Count`, `Truck_Count` | Counts by vehicle type |
| `Pedestrian_Count` | Pedestrian observations |
| `Congestion_Level` | Low / Medium / High |
| `Accident_Occurred` | Binary flag (0 = no, 1 = yes) |
| `Accident_Severity` | None / Minor / Moderate / Severe |
| `Emergency_Response_Time` | Time (minutes) for emergency response |
| `Traffic_Signal_Status` | Normal / Fault / Off |
| `Holiday`, `Special_Event` | Binary flags |
| `Air_Quality_Index` | AQI value |
| `Parking_Occupancy` | % parking occupancy |
| `Population_Density` | People per sq km |

### Dataset Link / Placeholder
Place your CSV file in the project root directory and upload it via the app's file uploader, **or** use the built-in synthetic data generator.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit ≥ 1.32 | Web dashboard framework |
| Pandas ≥ 2.0 | Data manipulation |
| NumPy ≥ 1.26 | Numerical computations |
| Plotly ≥ 5.18 | Interactive charts & maps |
| Matplotlib ≥ 3.8 | Static plots (heatmap) |
| Seaborn ≥ 0.13 | Statistical visualization |
| Scikit-learn ≥ 1.4 | Machine learning pipeline |

---

## 📁 Project Structure

```
TrafficPulse-AI/
│
├── MiroslavMandi_TrafficPulseAI.py              # Main Streamlit application (all-in-one)
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
├── MiroslavMandi_TrafficPulseAI_ProjectReport.docx  # Full project report
└── TrafficPulse_AI_Dataset.csv                  # Dataset (optional — upload via app)
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

```bash
# 1. Clone or download the project
git clone https://github.com/your-repo/TrafficPulse-AI.git
cd TrafficPulse-AI

# 2. (Recommended) Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run Command

```bash
streamlit run MiroslavMandi_TrafficPulseAI.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📊 Dashboard Features

### 🏠 Overview Page
- 7 KPI cards: Total Records, Traffic Volume, Accidents, Accident Rate, Avg Speed, High Congestion %, High-Risk Location
- Daily traffic volume time-series area chart
- Congestion level pie chart
- Automatic plain-language insights
- CSV download of filtered data

### 📈 Traffic Analysis Page
- Trends tab: Day-of-week bar chart, monthly line chart, cumulative area chart
- Peak Hours tab: Hourly traffic bar (peak hours highlighted), speed-by-hour line
- Vehicle Mix tab: Vehicle type pie chart, road-type bar chart
- Correlations tab: Full numeric correlation heatmap, traffic-vs-accident scatter

### ⚠️ Accident Analysis Page
- Accidents by weather, road condition, severity
- Location-level risk bar chart (horizontal)
- Hour-of-day and day-of-week accident charts

### 🚧 Congestion Analysis Page
- Congestion count bar, traffic density box plot, speed violin plot
- Weather × congestion stacked bar, speed × volume scatter
- Hour-of-day congestion stacked bar

### 🗺️ Map View Page
- Interactive Mapbox scatter map coloured by congestion level
- Accident density heatmap overlay

---

## 🤖 ML Features

| Task | Type | Models Available |
|---|---|---|
| Congestion Level Prediction | Multi-class classification | Random Forest, Gradient Boosting, Logistic Regression |
| Accident Occurrence Prediction | Binary classification | Random Forest, Gradient Boosting, Logistic Regression |

**Pipeline:** SimpleImputer → StandardScaler → Classifier

**Metrics reported:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, Full Classification Report

---

## 📤 Expected Output

After running, you will see:
- A dark-themed, multi-page Streamlit dashboard
- Live KPI metrics computed from the loaded/generated data
- Interactive Plotly charts across all analysis sections
- A trained ML model with performance metrics and feature importance chart
- Downloadable CSVs for filtered data and model results

---

## ⚠️ Limitations

1. **Synthetic data** does not capture real-world non-linear patterns with full fidelity.
2. ML models are trained on limited features; real-time sensor data would improve accuracy.
3. Map view requires valid `Latitude`/`Longitude` columns in the CSV.
4. No real-time data streaming; the app analyzes static snapshots.
5. Large datasets (>500 K rows) may slow down the Streamlit UI.

---

## 🚀 Future Improvements

- Real-time traffic data integration via REST APIs (e.g., TomTom, HERE)
- LSTM / Transformer-based time-series forecestion models
- Route optimization module with graph-based shortest-path suggestions
- Email/SMS alert system triggered by high-risk predictions
- Cloud deployment (Streamlit Community Cloud, AWS, GCP)
- Multi-language (i18n) support

---

## 👤 Author

**Miroslav Mandi**    
Year: 2026

---

## 📄 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute this project with attribution.

```
MIT License
Copyright (c) 2024 Miroslav Mandi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to use, copy, modify, merge, publish, or distribute it,
subject to the following conditions: the above copyright notice shall be
included in all copies or substantial portions of the Software.
```
