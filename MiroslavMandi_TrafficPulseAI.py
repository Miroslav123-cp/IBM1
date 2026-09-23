# =============================================================================
# TrafficPulse AI — Traffic, Congestion & Accident Analytics
# Author  : Miroslav Mandi
# Description : Complete Streamlit dashboard — data analysis, visualisation
#               & ML predictions for urban traffic data.
# Run     : streamlit run MiroslavMandi_TrafficPulseAI.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="TrafficPulse AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
/* ── global ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── KPI card ── */
.kpi-card {
    background: linear-gradient(135deg,#1e2130 0%,#252a3a 100%);
    border: 1px solid #2e3450;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,.4);
}
.kpi-title { color:#8892b0; font-size:12px; font-weight:600;
             text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
.kpi-value { color:#e6f1ff; font-size:26px; font-weight:700; line-height:1.1; }
.kpi-delta { font-size:12px; margin-top:4px; }
.kpi-delta.up   { color:#4ade80; }
.kpi-delta.down { color:#f87171; }
.kpi-delta.neu  { color:#94a3b8; }

/* ── section header ── */
.section-header {
    background: linear-gradient(90deg,#1e3a5f,#0f1117);
    border-left: 4px solid #4f8ef7;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 24px 0 14px;
    color: #e6f1ff;
    font-size: 17px;
    font-weight: 700;
}

/* ── insight box ── */
.insight-box {
    background: #1a2035;
    border: 1px solid #2e4070;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #ccd6f6;
    font-size: 14px;
    line-height: 1.6;
}
.insight-box ul { margin: 6px 0 0 18px; padding: 0; }

/* ── tab styling ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1e2130;
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    color: #8892b0;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#4f8ef7,#7c5cd8) !important;
    color: #fff !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1117 0%,#161b2e 100%);
    border-right: 1px solid #21262d;
}

/* ── upload area ── */
.upload-box {
    border: 2px dashed #4f8ef7;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: #8892b0;
    margin-bottom: 12px;
}

/* ── footer ── */
.footer {
    text-align: center;
    color: #4a5568;
    font-size: 12px;
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PLOTLY THEME HELPER  (applied via update_layout — NOT as px kwargs)
# =============================================================================
_BG = "#0f1117"
_SURFACE = "#1e2130"


def _theme(fig, height: int = 360):
    """Apply dark theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_SURFACE,
        font_color="#ccd6f6",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


COLOR_SEQ = px.colors.qualitative.Vivid

# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic traffic dataset (clearly labelled)."""
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=int(i)) for i in range(n)]

    df = pd.DataFrame()
    df["Record_ID"]   = range(1, n + 1)
    df["timestamp"]   = timestamps
    df["Date"]        = [t.date() for t in timestamps]
    df["Time"]        = [t.strftime("%H:%M:%S") for t in timestamps]
    df["Day_of_Week"] = [t.strftime("%A") for t in timestamps]
    df["Month"]       = [t.month for t in timestamps]
    df["Year"]        = [t.year for t in timestamps]
    df["hour"]        = [t.hour for t in timestamps]

    locations = ["Salt Lake","Jadavpur","Gariahat","New Town","Park Street",
                 "Howrah","Dumdum","Ballygunge","Ultadanga","Esplanade"]
    lats = {"Salt Lake":22.57,"Jadavpur":22.49,"Gariahat":22.51,"New Town":22.59,
            "Park Street":22.55,"Howrah":22.58,"Dumdum":22.65,"Ballygunge":22.53,
            "Ultadanga":22.58,"Esplanade":22.56}
    lons = {"Salt Lake":88.41,"Jadavpur":88.37,"Gariahat":88.36,"New Town":88.47,
            "Park Street":88.35,"Howrah":88.31,"Dumdum":88.41,"Ballygunge":88.36,
            "Ultadanga":88.39,"Esplanade":88.35}
    df["Location"]  = rng.choice(locations, n)
    df["Latitude"]  = df["Location"].map(lats) + rng.uniform(-0.01, 0.01, n)
    df["Longitude"] = df["Location"].map(lons) + rng.uniform(-0.01, 0.01, n)

    df["Road_Type"]      = rng.choice(["Arterial","Highway","Urban","Residential","Expressway"],
                                       n, p=[0.30, 0.25, 0.25, 0.15, 0.05])
    df["Road_Condition"] = rng.choice(["Good","Fair","Poor"], n, p=[0.50, 0.35, 0.15])
    df["Weather"]        = rng.choice(["Clear","Rain","Fog","Cloudy","Storm"],
                                       n, p=[0.45, 0.25, 0.15, 0.10, 0.05])

    df["Temperature_C"] = rng.normal(25, 6, n).clip(5, 45).round(1)
    df["Visibility_km"] = np.where(df["Weather"] == "Fog",
                                   rng.uniform(0.5, 3, n),
                                   rng.uniform(5, 10, n)).round(2)
    df["Rainfall"]      = np.where(df["Weather"].isin(["Rain","Storm"]),
                                   rng.uniform(0.1, 30, n), 0.0).round(2)

    peak = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    base_vol = 200 + peak * 400 + rng.normal(0, 50, n)
    df["Traffic_Volume"]    = base_vol.clip(10).astype(int)
    df["Vehicle_Count"]     = (base_vol * rng.uniform(0.6, 1.0, n)).clip(1).astype(int)
    df["Car_Count"]         = (df["Vehicle_Count"] * rng.uniform(0.40, 0.60, n)).astype(int)
    df["Bike_Count"]        = (df["Vehicle_Count"] * rng.uniform(0.10, 0.20, n)).astype(int)
    df["Bus_Count"]         = (df["Vehicle_Count"] * rng.uniform(0.05, 0.15, n)).astype(int)
    df["Truck_Count"]       = (df["Vehicle_Count"] * rng.uniform(0.03, 0.08, n)).astype(int)
    df["Pedestrian_Count"]  = (rng.poisson(30, n) + peak * 40).clip(0).astype(int)

    speed_base = 60 - (df["Traffic_Volume"] / 20).clip(0, 40) + rng.normal(0, 5, n)
    df["Average_Speed_kmph"] = speed_base.clip(5, 100).round(1)
    df["Traffic_Density"]    = (df["Vehicle_Count"] / 10).round(2)
    df["Travel_Time"]        = rng.uniform(3, 20, n).round(2)

    df["Congestion_Level"] = pd.cut(
        df["Traffic_Volume"],
        bins=[-np.inf, 250, 500, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    acc_prob = (
        0.03
        + 0.04 * (df["Weather"] == "Fog").astype(int)
        + 0.03 * (df["Weather"] == "Storm").astype(int)
        + 0.02 * (df["Road_Condition"] == "Poor").astype(int)
    )
    df["Accident_Occurred"] = (rng.uniform(0, 1, n) < acc_prob).astype(int)
    df["Accident_Severity"] = np.where(
        df["Accident_Occurred"] == 0, "None",
        rng.choice(["Minor","Moderate","Severe"], n, p=[0.50, 0.35, 0.15]),
    )
    df["Emergency_Response_Time"] = np.where(
        df["Accident_Occurred"] == 1, rng.uniform(5, 40, n).round(1), np.nan
    )

    df["Traffic_Signal_Status"] = rng.choice(["Normal","Fault","Off"], n,
                                              p=[0.85, 0.10, 0.05])
    df["Holiday"]       = rng.choice([0, 1], n, p=[0.93, 0.07])
    df["Special_Event"] = rng.choice([0, 1], n, p=[0.95, 0.05])
    df["Air_Quality_Index"]   = rng.integers(30, 300, n)
    df["Parking_Occupancy"]   = rng.uniform(10, 95, n).round(2)
    df["Population_Density"]  = rng.integers(5000, 20000, n)
    return df


# =============================================================================
# DATA LOADER
# =============================================================================
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names from any source CSV to the project schema."""
    rename = {
        "id": "Record_ID", "timestamp": "timestamp",
        "date": "Date", "time": "Time", "day_of_week": "Day_of_Week",
        "month": "Month", "hour": "hour", "location": "Location",
        "road_type": "Road_Type", "vehicle_count": "Vehicle_Count",
        "average_speed": "Average_Speed_kmph", "traffic_density": "Traffic_Density",
        "travel_time": "Travel_Time", "weather": "Weather",
        "temperature": "Temperature_C", "rainfall": "Rainfall",
        "visibility": "Visibility_km", "accident": "Accident_Occurred",
        "road_condition": "Road_Condition", "signal_status": "Traffic_Signal_Status",
        "parking_occupancy": "Parking_Occupancy",
        "population_density": "Population_Density",
        "congestion_level": "Congestion_Level",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "Traffic_Volume" not in df.columns and "Vehicle_Count" in df.columns:
        df["Traffic_Volume"] = df["Vehicle_Count"]
    if "Year" not in df.columns and "Date" in df.columns:
        df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    if "Accident_Severity" not in df.columns and "Accident_Occurred" in df.columns:
        df["Accident_Severity"] = np.where(df["Accident_Occurred"] == 1, "Minor", "None")
    for col in ["Latitude","Longitude","Car_Count","Bike_Count","Bus_Count",
                "Truck_Count","Pedestrian_Count","Air_Quality_Index",
                "Holiday","Special_Event","Emergency_Response_Time"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, validate, impute, clip and engineer features."""
    df = df.copy()

    # Date parsing
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Derive hour from Date if missing
    if "hour" not in df.columns and "Date" in df.columns:
        df["hour"] = df["Date"].dt.hour

    # Numeric coerce
    num_cols = ["Traffic_Volume","Vehicle_Count","Average_Speed_kmph",
                "Temperature_C","Visibility_km","Accident_Occurred",
                "Latitude","Longitude","Air_Quality_Index","Parking_Occupancy",
                "Population_Density","Traffic_Density","Travel_Time","Rainfall",
                "Car_Count","Bike_Count","Bus_Count","Truck_Count","Pedestrian_Count",
                "hour","Month","Year"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Duplicates
    df.drop_duplicates(inplace=True)

    # Missing value imputation
    for c in ["Traffic_Volume","Vehicle_Count","Average_Speed_kmph"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in ["Weather","Road_Condition","Road_Type","Congestion_Level"]:
        if c in df.columns:
            mode = df[c].mode()
            df[c] = df[c].fillna(mode[0] if len(mode) else "Unknown")

    # Outlier clipping
    if "Average_Speed_kmph" in df.columns:
        df["Average_Speed_kmph"] = df["Average_Speed_kmph"].clip(0, 150)
    if "Traffic_Volume" in df.columns:
        df["Traffic_Volume"] = df["Traffic_Volume"].clip(0)

    # Feature engineering
    if "Date" in df.columns and df["Date"].notna().any():
        df["Month_Name"]   = df["Date"].dt.month_name()
        df["Week_Number"]  = df["Date"].dt.isocalendar().week.astype("Int64")
    if "hour" in df.columns:
        df["Is_Peak_Hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        bins   = [-1, 5, 11, 16, 20, 24]
        labels = ["Night","Morning","Afternoon","Evening","Night"]
        df["Time_Period"] = pd.cut(df["hour"], bins=bins,
                                   labels=labels[:4] + ["Night"],
                                   right=True, ordered=False).astype(str)
    if "Traffic_Volume" in df.columns and "Vehicle_Count" in df.columns:
        df["Vehicles_per_Volume"] = (
            df["Vehicle_Count"] / df["Traffic_Volume"].replace(0, np.nan)
        ).round(3)
    if "Accident_Occurred" in df.columns and "Traffic_Volume" in df.columns:
        df["Accident_Rate"] = (
            df["Accident_Occurred"] / df["Traffic_Volume"].replace(0, np.nan) * 100
        ).round(4)
    return df


# =============================================================================
# UI HELPERS
# =============================================================================

def kpi_card(title: str, value: str, delta: str = "", delta_dir: str = "neu") -> str:
    delta_html = (f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else "")
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}</div>'
    )


def section_header(title: str, icon: str = "📊") -> None:
    st.markdown(f'<div class="section-header">{icon} {title}</div>',
                unsafe_allow_html=True)


def insight_box(bullets: list) -> None:
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f'<div class="insight-box"><ul>{items}</ul></div>',
                unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

def build_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🚦 TrafficPulse AI")
        st.markdown("---")
        page = st.radio("Navigation", [
            "🏠 Overview",
            "📈 Traffic Analysis",
            "⚠️ Accident Analysis",
            "🚧 Congestion Analysis",
            "🗺️ Map View",
            "🤖 ML Predictions",
            "ℹ️ About Project",
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### 🔍 Filters")

        date_range = None
        if "Date" in df.columns and df["Date"].notna().any():
            min_d = df["Date"].min()
            max_d = df["Date"].max()
            if pd.notna(min_d) and pd.notna(max_d):
                try:
                    date_range = st.date_input(
                        "Date Range",
                        value=[min_d.date(), max_d.date()],
                        min_value=min_d.date(), max_value=max_d.date(),
                    )
                except Exception:
                    date_range = None

        locs     = sorted(df["Location"].dropna().unique().tolist())    if "Location"      in df.columns else []
        weathers = sorted(df["Weather"].dropna().unique().tolist())     if "Weather"       in df.columns else []
        rtypes   = sorted(df["Road_Type"].dropna().unique().tolist())   if "Road_Type"     in df.columns else []
        rconds   = sorted(df["Road_Condition"].dropna().unique().tolist()) if "Road_Condition" in df.columns else []
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        avail_days = [d for d in day_order if d in df.get("Day_of_Week", pd.Series()).values] if "Day_of_Week" in df.columns else []
        cong_opts  = sorted(df["Congestion_Level"].dropna().unique().tolist()) if "Congestion_Level" in df.columns else []

        sel_loc    = st.multiselect("Location",      locs,       default=[])
        sel_weather = st.multiselect("Weather",      weathers,   default=[])
        sel_rtype  = st.multiselect("Road Type",     rtypes,     default=[])
        sel_rcond  = st.multiselect("Road Condition",rconds,     default=[])
        sel_day    = st.multiselect("Day of Week",   avail_days, default=[])
        sel_cong   = st.multiselect("Congestion",    cong_opts,  default=[])

        st.markdown("---")
        st.markdown(
            '<div class="footer">TrafficPulse AI v1.0<br>Miroslav Mandi © 2024</div>',
            unsafe_allow_html=True,
        )

    return page, date_range, sel_loc, sel_weather, sel_rtype, sel_rcond, sel_day, sel_cong


def apply_filters(df, date_range, sel_loc, sel_weather,
                  sel_rtype, sel_rcond, sel_day, sel_cong):
    fdf = df.copy()
    if date_range and len(date_range) == 2 and "Date" in fdf.columns:
        try:
            fdf = fdf[(fdf["Date"].dt.date >= date_range[0]) &
                      (fdf["Date"].dt.date <= date_range[1])]
        except Exception:
            pass
    if sel_loc    and "Location"       in fdf.columns: fdf = fdf[fdf["Location"].isin(sel_loc)]
    if sel_weather and "Weather"       in fdf.columns: fdf = fdf[fdf["Weather"].isin(sel_weather)]
    if sel_rtype  and "Road_Type"      in fdf.columns: fdf = fdf[fdf["Road_Type"].isin(sel_rtype)]
    if sel_rcond  and "Road_Condition" in fdf.columns: fdf = fdf[fdf["Road_Condition"].isin(sel_rcond)]
    if sel_day    and "Day_of_Week"    in fdf.columns: fdf = fdf[fdf["Day_of_Week"].isin(sel_day)]
    if sel_cong   and "Congestion_Level" in fdf.columns: fdf = fdf[fdf["Congestion_Level"].isin(sel_cong)]
    return fdf if len(fdf) > 0 else df


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

def page_overview(df: pd.DataFrame):
    section_header("Dashboard Overview", "🏠")

    total_rec = len(df)
    total_vol = int(df["Traffic_Volume"].sum())          if "Traffic_Volume"    in df.columns else 0
    total_acc = int(df["Accident_Occurred"].sum())       if "Accident_Occurred" in df.columns else 0
    acc_rate  = round(total_acc / total_rec * 100, 2)   if total_rec else 0
    avg_speed = round(df["Average_Speed_kmph"].mean(), 1) if "Average_Speed_kmph" in df.columns else 0

    cong_high_pct = 0
    if "Congestion_Level" in df.columns:
        cong_high_pct = round((df["Congestion_Level"] == "High").mean() * 100, 1)

    high_risk = "N/A"
    if "Location" in df.columns and "Accident_Occurred" in df.columns:
        hr = df.groupby("Location")["Accident_Occurred"].sum()
        if len(hr):
            high_risk = str(hr.idxmax())

    cols = st.columns(7)
    cards = [
        ("Total Records",       f"{total_rec:,}",        "", "neu"),
        ("Traffic Volume",      f"{total_vol:,}",        "", "neu"),
        ("Accidents",           f"{total_acc:,}",        "", "down"),
        ("Accident Rate",       f"{acc_rate}%",          "", "down"),
        ("Avg Speed (km/h)",    f"{avg_speed}",          "", "neu"),
        ("High Congestion",     f"{cong_high_pct}%",     "", "down"),
        ("High-Risk Location",  high_risk,               "", "down"),
    ]
    for col, (title, val, delta, ddir) in zip(cols, cards):
        col.markdown(kpi_card(title, val, delta, ddir), unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        section_header("Traffic Volume Over Time", "📈")
        if "Date" in df.columns and "Traffic_Volume" in df.columns:
            ts = (df.groupby("Date")["Traffic_Volume"].sum()
                    .reset_index().sort_values("Date"))
            fig = px.area(ts, x="Date", y="Traffic_Volume",
                          title="Daily Traffic Volume",
                          color_discrete_sequence=["#4f8ef7"])
            st.plotly_chart(_theme(fig, 320), use_container_width=True)

    with col2:
        section_header("Congestion Level Distribution", "🚧")
        if "Congestion_Level" in df.columns:
            cong_cnt = df["Congestion_Level"].value_counts().reset_index()
            cong_cnt.columns = ["Level", "Count"]
            fig = px.pie(cong_cnt, names="Level", values="Count",
                         color="Level", title="Congestion Levels",
                         color_discrete_map={"Low":"#4ade80","Medium":"#fbbf24","High":"#f87171"})
            st.plotly_chart(_theme(fig, 320), use_container_width=True)

    # Automatic plain-language insights
    section_header("Automatic Insights", "💡")
    insights = []
    if "Weather" in df.columns and "Accident_Occurred" in df.columns:
        worst_w = df.groupby("Weather")["Accident_Occurred"].mean()
        if len(worst_w):
            insights.append(f"<b>{worst_w.idxmax()}</b> weather has the highest accident probability "
                            f"({worst_w.max()*100:.1f}%).")
    if "hour" in df.columns and "Traffic_Volume" in df.columns:
        ph = df.groupby("hour")["Traffic_Volume"].mean()
        if len(ph):
            insights.append(f"Peak traffic hour is <b>{int(ph.idxmax())}:00</b> "
                            f"(avg volume {ph.max():.0f}).")
    if "Road_Condition" in df.columns and "Accident_Occurred" in df.columns:
        worst_rc = df.groupby("Road_Condition")["Accident_Occurred"].mean()
        if len(worst_rc):
            insights.append(f"<b>{worst_rc.idxmax()}</b> road condition has the highest accident rate.")
    if "Congestion_Level" in df.columns:
        insights.append(f"<b>{cong_high_pct}%</b> of observations show High congestion level.")
    if "Average_Speed_kmph" in df.columns:
        q25 = df["Average_Speed_kmph"].quantile(0.25)
        insights.append(f"25% of observations recorded speeds below <b>{q25:.1f} km/h</b>.")
    if "Location" in df.columns and "Accident_Occurred" in df.columns and high_risk != "N/A":
        insights.append(f"<b>{high_risk}</b> is the highest-risk location by accident count.")
    insight_box(insights if insights else ["Load data to generate automatic insights."])

    # Download
    section_header("Download Filtered Data", "⬇️")
    st.download_button(
        "⬇️ Download Filtered Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="TrafficPulse_filtered.csv",
        mime="text/csv",
    )


# =============================================================================
# PAGE: TRAFFIC ANALYSIS
# =============================================================================

def page_traffic(df: pd.DataFrame):
    section_header("Traffic Analysis", "📈")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📅 Trends", "⏰ Peak Hours", "🚗 Vehicle Mix", "🔗 Correlations"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if "Day_of_Week" in df.columns and "Traffic_Volume" in df.columns:
                day_order = ["Monday","Tuesday","Wednesday","Thursday",
                             "Friday","Saturday","Sunday"]
                dv = (df.groupby("Day_of_Week")["Traffic_Volume"].mean()
                        .reindex([d for d in day_order if d in df["Day_of_Week"].values])
                        .reset_index())
                dv.columns = ["Day", "Avg Traffic Volume"]
                fig = px.bar(dv, x="Day", y="Avg Traffic Volume",
                             color="Avg Traffic Volume", color_continuous_scale="Viridis",
                             title="Avg Traffic by Day of Week")
                st.plotly_chart(_theme(fig, 340), use_container_width=True)

        with col2:
            if "Month" in df.columns and "Traffic_Volume" in df.columns:
                mv = (df.groupby("Month")["Traffic_Volume"].mean()
                        .reset_index().sort_values("Month"))
                mv.columns = ["Month", "Avg Traffic Volume"]
                fig = px.line(mv, x="Month", y="Avg Traffic Volume",
                              markers=True, color_discrete_sequence=["#f97316"],
                              title="Avg Traffic by Month")
                st.plotly_chart(_theme(fig, 340), use_container_width=True)

        if "Date" in df.columns and "Traffic_Volume" in df.columns:
            ts = (df.groupby("Date")["Traffic_Volume"].sum()
                    .reset_index().sort_values("Date"))
            fig = px.area(ts, x="Date", y="Traffic_Volume",
                          color_discrete_sequence=["#4f8ef7"],
                          title="Total Traffic Volume Over Time")
            st.plotly_chart(_theme(fig, 300), use_container_width=True)

    with tab2:
        if "hour" in df.columns and "Traffic_Volume" in df.columns:
            hv = df.groupby("hour")["Traffic_Volume"].mean().reset_index()
            hv.columns = ["Hour", "Avg Traffic Volume"]
            hv["Is_Peak"] = hv["Hour"].isin([7, 8, 9, 17, 18, 19])
            fig = px.bar(hv, x="Hour", y="Avg Traffic Volume",
                         color="Is_Peak",
                         color_discrete_map={True: "#f97316", False: "#4f8ef7"},
                         title="Avg Traffic by Hour  (orange = peak hours)")
            st.plotly_chart(_theme(fig, 360), use_container_width=True)

        if "hour" in df.columns and "Average_Speed_kmph" in df.columns:
            hs = df.groupby("hour")["Average_Speed_kmph"].mean().reset_index()
            hs.columns = ["Hour", "Avg Speed (km/h)"]
            fig = px.line(hs, x="Hour", y="Avg Speed (km/h)", markers=True,
                          color_discrete_sequence=["#a78bfa"],
                          title="Average Speed by Hour of Day")
            st.plotly_chart(_theme(fig, 300), use_container_width=True)

    with tab3:
        vcols = [c for c in ["Car_Count","Bike_Count","Bus_Count","Truck_Count",
                              "Pedestrian_Count"]
                 if c in df.columns and df[c].notna().any()]
        if vcols:
            vmeans = df[vcols].mean().reset_index()
            vmeans.columns = ["Vehicle Type", "Avg Count"]
            vmeans["Vehicle Type"] = vmeans["Vehicle Type"].str.replace("_Count", "")
            fig = px.pie(vmeans, names="Vehicle Type", values="Avg Count",
                         color_discrete_sequence=COLOR_SEQ,
                         title="Vehicle Type Distribution")
            st.plotly_chart(_theme(fig, 360), use_container_width=True)

        if "Road_Type" in df.columns and "Traffic_Volume" in df.columns:
            rt = df.groupby("Road_Type")["Traffic_Volume"].mean().reset_index()
            rt.columns = ["Road Type", "Avg Traffic"]
            fig = px.bar(rt, x="Road Type", y="Avg Traffic",
                         color="Road Type", color_discrete_sequence=COLOR_SEQ,
                         title="Avg Traffic by Road Type")
            st.plotly_chart(_theme(fig, 340), use_container_width=True)

    with tab4:
        num_df = df.select_dtypes(include=np.number).drop(
            columns=["Record_ID"] if "Record_ID" in df.columns else [], errors="ignore"
        )
        valid = num_df.dropna(axis=1, how="all")
        if len(valid.columns) > 2:
            corr = valid.corr()
            fig_h, ax = plt.subplots(figsize=(12, 9))
            fig_h.patch.set_facecolor(_BG)
            ax.set_facecolor(_SURFACE)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        linewidths=0.5, ax=ax, annot_kws={"size": 7},
                        cbar_kws={"shrink": 0.8})
            ax.tick_params(colors="#ccd6f6", labelsize=7)
            plt.title("Correlation Heatmap", color="#ccd6f6", fontsize=13, pad=10)
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

        if "Traffic_Volume" in df.columns and "Accident_Occurred" in df.columns:
            samp = df.sample(min(2000, len(df)), random_state=42)
            fig = px.scatter(
                samp, x="Traffic_Volume", y="Accident_Occurred",
                color="Congestion_Level" if "Congestion_Level" in df.columns else None,
                opacity=0.45, title="Traffic Volume vs Accident Occurred",
                color_discrete_map={"Low":"#4ade80","Medium":"#fbbf24","High":"#f87171"},
            )
            st.plotly_chart(_theme(fig, 360), use_container_width=True)


# =============================================================================
# PAGE: ACCIDENT ANALYSIS
# =============================================================================

def page_accident(df: pd.DataFrame):
    section_header("Accident Analysis", "⚠️")

    total_acc = int(df["Accident_Occurred"].sum()) if "Accident_Occurred" in df.columns else 0
    acc_rate  = round(total_acc / len(df) * 100, 2) if len(df) else 0
    severe    = int((df["Accident_Severity"] == "Severe").sum()) if "Accident_Severity" in df.columns else 0
    avg_rt    = round(df["Emergency_Response_Time"].dropna().mean(), 1) \
                if "Emergency_Response_Time" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Total Accidents",      f"{total_acc:,}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Accident Rate",        f"{acc_rate}%"),  unsafe_allow_html=True)
    c3.markdown(kpi_card("Severe Accidents",     f"{severe:,}"),   unsafe_allow_html=True)
    c4.markdown(kpi_card("Avg Response (min)",   f"{avg_rt}"),     unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌦️ Weather & Road", "📍 Location", "⏰ Time Patterns"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if "Weather" in df.columns and "Accident_Occurred" in df.columns:
                wa = (df.groupby("Weather")["Accident_Occurred"].sum()
                        .reset_index().rename(columns={"Accident_Occurred":"Accidents"})
                        .sort_values("Accidents", ascending=False))
                fig = px.bar(wa, x="Weather", y="Accidents", color="Weather",
                             color_discrete_sequence=COLOR_SEQ,
                             title="Accidents by Weather Condition")
                fig.update_layout(showlegend=False)
                st.plotly_chart(_theme(fig, 340), use_container_width=True)

        with col2:
            if "Road_Condition" in df.columns and "Accident_Occurred" in df.columns:
                ra = (df.groupby("Road_Condition")["Accident_Occurred"].sum()
                        .reset_index().rename(columns={"Accident_Occurred":"Accidents"}))
                fig = px.bar(ra, x="Road_Condition", y="Accidents",
                             color="Road_Condition", color_discrete_sequence=COLOR_SEQ,
                             title="Accidents by Road Condition")
                fig.update_layout(showlegend=False)
                st.plotly_chart(_theme(fig, 340), use_container_width=True)

        if "Accident_Severity" in df.columns:
            sev = (df[df.get("Accident_Occurred", pd.Series(0, index=df.index)) == 1]
                   ["Accident_Severity"].value_counts().reset_index())
            sev.columns = ["Severity", "Count"]
            fig = px.pie(sev, names="Severity", values="Count",
                         color="Severity",
                         color_discrete_map={"Minor":"#fbbf24","Moderate":"#fb923c","Severe":"#f87171"},
                         title="Accident Severity Distribution")
            st.plotly_chart(_theme(fig, 340), use_container_width=True)

    with tab2:
        if "Location" in df.columns and "Accident_Occurred" in df.columns:
            la = (df.groupby("Location")["Accident_Occurred"].sum()
                    .reset_index().rename(columns={"Accident_Occurred":"Accidents"})
                    .sort_values("Accidents", ascending=True))
            fig = px.bar(la, x="Accidents", y="Location", orientation="h",
                         color="Accidents", color_continuous_scale="Reds",
                         title="Accidents by Location (High-Risk Ranking)")
            st.plotly_chart(_theme(fig, 420), use_container_width=True)

    with tab3:
        if "hour" in df.columns and "Accident_Occurred" in df.columns:
            ha = (df.groupby("hour")["Accident_Occurred"].sum()
                    .reset_index().rename(columns={"Accident_Occurred":"Accidents"}))
            fig = px.bar(ha, x="hour", y="Accidents", color="Accidents",
                         color_continuous_scale="OrRd",
                         title="Accidents by Hour of Day")
            st.plotly_chart(_theme(fig, 340), use_container_width=True)

        if "Day_of_Week" in df.columns and "Accident_Occurred" in df.columns:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            da = (df.groupby("Day_of_Week")["Accident_Occurred"].sum()
                    .reindex([d for d in day_order if d in df["Day_of_Week"].values])
                    .reset_index().rename(columns={"Accident_Occurred":"Accidents"}))
            fig = px.bar(da, x="Day_of_Week", y="Accidents",
                         color="Day_of_Week", color_discrete_sequence=COLOR_SEQ,
                         title="Accidents by Day of Week")
            fig.update_layout(showlegend=False)
            st.plotly_chart(_theme(fig, 340), use_container_width=True)


# =============================================================================
# PAGE: CONGESTION ANALYSIS
# =============================================================================

def page_congestion(df: pd.DataFrame):
    section_header("Congestion Analysis", "🚧")
    tab1, tab2 = st.tabs(["📊 Distributions", "🔬 Factor Analysis"])

    CMAP = {"Low":"#4ade80","Medium":"#fbbf24","High":"#f87171"}

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if "Congestion_Level" in df.columns:
                c = df["Congestion_Level"].value_counts().reset_index()
                c.columns = ["Level", "Count"]
                fig = px.bar(c, x="Level", y="Count", color="Level",
                             color_discrete_map=CMAP, title="Congestion Level Counts")
                fig.update_layout(showlegend=False)
                st.plotly_chart(_theme(fig, 340), use_container_width=True)

        with col2:
            if "Traffic_Density" in df.columns and "Congestion_Level" in df.columns:
                fig = px.box(df, x="Congestion_Level", y="Traffic_Density",
                             color="Congestion_Level", color_discrete_map=CMAP,
                             title="Traffic Density by Congestion Level")
                fig.update_layout(showlegend=False)
                st.plotly_chart(_theme(fig, 340), use_container_width=True)

        if "Average_Speed_kmph" in df.columns and "Congestion_Level" in df.columns:
            fig = px.violin(df, y="Average_Speed_kmph", color="Congestion_Level",
                            box=True, points=False, color_discrete_map=CMAP,
                            title="Speed Distribution by Congestion Level")
            st.plotly_chart(_theme(fig, 360), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if "Weather" in df.columns and "Congestion_Level" in df.columns:
                wc = df.groupby(["Weather","Congestion_Level"]).size().reset_index(name="Count")
                fig = px.bar(wc, x="Weather", y="Count", color="Congestion_Level",
                             barmode="stack", color_discrete_map=CMAP,
                             title="Weather vs Congestion Level")
                st.plotly_chart(_theme(fig, 360), use_container_width=True)

        with col2:
            if "Average_Speed_kmph" in df.columns and "Traffic_Volume" in df.columns:
                samp = df.sample(min(3000, len(df)), random_state=42)
                fig = px.scatter(samp, x="Average_Speed_kmph", y="Traffic_Volume",
                                 color="Congestion_Level" if "Congestion_Level" in df.columns else None,
                                 opacity=0.45, color_discrete_map=CMAP,
                                 title="Speed vs Traffic Volume")
                st.plotly_chart(_theme(fig, 360), use_container_width=True)

        if "hour" in df.columns and "Congestion_Level" in df.columns:
            hc = df.groupby(["hour","Congestion_Level"]).size().reset_index(name="Count")
            fig = px.bar(hc, x="hour", y="Count", color="Congestion_Level",
                         barmode="stack", color_discrete_map=CMAP,
                         title="Congestion Level by Hour of Day")
            st.plotly_chart(_theme(fig, 360), use_container_width=True)


# =============================================================================
# PAGE: MAP VIEW
# =============================================================================

def page_map(df: pd.DataFrame):
    section_header("Interactive Map View", "🗺️")

    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        st.info("📍 No geographic coordinates in this dataset. "
                "Switch to Synthetic Dataset mode to enable the map.")
        return

    map_df = df.dropna(subset=["Latitude","Longitude"]).copy()
    map_df = map_df[(map_df["Latitude"].between(-90, 90)) &
                    (map_df["Longitude"].between(-180, 180))]
    if len(map_df) == 0:
        st.warning("No valid coordinate records found.")
        return

    color_col = "Congestion_Level" if "Congestion_Level" in map_df.columns else None
    samp = map_df.sample(min(3000, len(map_df)), random_state=42)

    hover = {c: True for c in ["Location","Weather","Traffic_Volume",
                                "Average_Speed_kmph","Accident_Occurred"]
             if c in samp.columns}

    # Use px.scatter_map (Plotly ≥ 6) with fallback to scatter_mapbox (Plotly < 6)
    try:
        fig = px.scatter_map(
            samp, lat="Latitude", lon="Longitude",
            color=color_col,
            color_discrete_map={"Low":"#4ade80","Medium":"#fbbf24","High":"#f87171"},
            hover_data=hover,
            zoom=11, map_style="carto-darkmatter",
            title="Traffic Congestion Map",
        )
    except AttributeError:
        fig = px.scatter_mapbox(
            samp, lat="Latitude", lon="Longitude",
            color=color_col,
            color_discrete_map={"Low":"#4ade80","Medium":"#fbbf24","High":"#f87171"},
            hover_data=hover,
            zoom=11, mapbox_style="carto-darkmatter",
            title="Traffic Congestion Map",
        )

    fig.update_layout(
        paper_bgcolor=_BG, font_color="#ccd6f6", height=550,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accident hotspot density
    if "Accident_Occurred" in map_df.columns:
        acc_map = map_df[map_df["Accident_Occurred"] == 1]
        if len(acc_map) > 0:
            section_header("Accident Hotspot Density", "🔥")
            try:
                fig2 = px.density_map(
                    acc_map, lat="Latitude", lon="Longitude",
                    z="Accident_Occurred", radius=18, zoom=11,
                    map_style="carto-darkmatter",
                    color_continuous_scale="Hot",
                    title="Accident Density Heatmap",
                )
            except AttributeError:
                fig2 = px.density_mapbox(
                    acc_map, lat="Latitude", lon="Longitude",
                    z="Accident_Occurred", radius=18, zoom=11,
                    mapbox_style="carto-darkmatter",
                    color_continuous_scale="Hot",
                    title="Accident Density Heatmap",
                )
            fig2.update_layout(
                paper_bgcolor=_BG, font_color="#ccd6f6", height=500,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PAGE: ML PREDICTIONS
# =============================================================================

def page_ml(df: pd.DataFrame):
    section_header("Machine Learning Predictions", "🤖")
    st.markdown("""
    <div class="insight-box">
    Two ML tasks are available:<br>
    <b>1. Congestion Level Prediction</b> — Multi-class (Low / Medium / High)<br>
    <b>2. Accident Occurrence Prediction</b> — Binary (0 / 1)
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        task = st.selectbox("Task", [
            "Congestion Level Prediction",
            "Accident Occurrence Prediction",
        ])
    with col2:
        model_name = st.selectbox("Model", [
            "Random Forest",
            "Gradient Boosting",
            "Logistic Regression",
        ])
    with col3:
        test_pct = st.slider("Test Split %", 10, 40, 20)

    if st.button("🚀 Train & Evaluate Model", type="primary"):
        with st.spinner("Training model — please wait…"):
            _run_ml(df, task, model_name, test_pct / 100)


def _run_ml(df, task, model_name, test_size):
    feature_cols = [c for c in [
        "hour","Month","Temperature_C","Visibility_km","Traffic_Volume",
        "Average_Speed_kmph","Vehicle_Count","Rainfall","Traffic_Density",
        "Travel_Time","Parking_Occupancy","Population_Density",
        "Air_Quality_Index","Is_Peak_Hour",
        "Weather","Road_Condition","Road_Type","Day_of_Week",
        "Traffic_Signal_Status",
    ] if c in df.columns]

    target_col = ("Congestion_Level" if task == "Congestion Level Prediction"
                  else "Accident_Occurred")

    if target_col not in df.columns:
        st.error(f"Column '{target_col}' not found."); return

    ml_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    if len(ml_df) < 100:
        st.error("Need at least 100 rows after cleaning."); return

    # Encode categoricals
    cat_cols = [c for c in ml_df.select_dtypes(include="object").columns
                if c != target_col]
    for c in cat_cols:
        ml_df[c] = LabelEncoder().fit_transform(ml_df[c].astype(str))

    le_target = None
    if ml_df[target_col].dtype == object:
        le_target = LabelEncoder()
        ml_df[target_col] = le_target.fit_transform(ml_df[target_col].astype(str))

    X, y = ml_df[feature_cols], ml_df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        clf = LogisticRegression(max_iter=500, random_state=42, multi_class="auto")

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler()),
                     ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        proba = pipe.predict_proba(X_test)
        roc = (roc_auc_score(y_test, proba[:, 1])
               if len(np.unique(y)) == 2
               else roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        roc_str = f"{roc:.4f}"
    except Exception:
        roc_str = "N/A"

    st.success("✅ Model trained successfully!")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(kpi_card("Accuracy",  f"{acc:.4f}"),  unsafe_allow_html=True)
    m2.markdown(kpi_card("Precision", f"{prec:.4f}"), unsafe_allow_html=True)
    m3.markdown(kpi_card("Recall",    f"{rec:.4f}"),  unsafe_allow_html=True)
    m4.markdown(kpi_card("F1-Score",  f"{f1:.4f}"),   unsafe_allow_html=True)
    m5.markdown(kpi_card("ROC-AUC",   roc_str),       unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_header("Confusion Matrix", "🔢")
        labels = le_target.classes_ if le_target is not None else None
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor(_BG)
        ax.set_facecolor(_SURFACE)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.tick_params(colors="#ccd6f6")
        ax.set_title("Confusion Matrix", color="#ccd6f6")
        for txt in ax.texts:
            txt.set_color("#0f1117")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col2:
        section_header("Feature Importances", "📊")
        if hasattr(clf, "feature_importances_"):
            fi = (pd.DataFrame({"Feature": feature_cols,
                                "Importance": clf.feature_importances_})
                    .sort_values("Importance", ascending=False).head(15))
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Viridis",
                         title="Top Feature Importances")
            fig.update_layout(yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(_theme(fig, 420), use_container_width=True)
        elif hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_).mean(axis=0)
            fi = (pd.DataFrame({"Feature": feature_cols, "Coefficient": coefs})
                    .sort_values("Coefficient", ascending=False).head(15))
            fig = px.bar(fi, x="Coefficient", y="Feature", orientation="h",
                         color="Coefficient", color_continuous_scale="Viridis",
                         title="Feature Coefficients")
            fig.update_layout(yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(_theme(fig, 420), use_container_width=True)

    section_header("Detailed Classification Report", "📋")
    label_names = ([str(l) for l in le_target.classes_] if le_target is not None
                   else [str(v) for v in sorted(y.unique())])
    report = classification_report(y_test, y_pred,
                                   target_names=label_names,
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    st.download_button(
        "⬇️ Download Classification Report (CSV)",
        data=pd.DataFrame(report).transpose().round(4).to_csv().encode("utf-8"),
        file_name=f"TrafficPulse_ML_{model_name.replace(' ','_')}.csv",
        mime="text/csv",
    )


# =============================================================================
# PAGE: ABOUT PROJECT
# =============================================================================

def page_about():
    section_header("About TrafficPulse AI", "ℹ️")
    st.markdown("""
    <div class="insight-box">
    <b>🎯 Project Objectives</b>
    <ul>
        <li>Analyse urban traffic patterns, congestion levels, and accident occurrences.</li>
        <li>Identify high-risk locations, peak hours, and weather-related risk factors.</li>
        <li>Provide actionable insights through interactive visualisations.</li>
        <li>Apply ML models to predict congestion level and accident occurrence.</li>
    </ul>
    </div>

    <div class="insight-box">
    <b>🗄️ Dataset</b>
    <ul>
        <li><b>Source:</b> Real CSV upload OR auto-generated synthetic data.</li>
        <li><b>Key fields:</b> Date, Time, Location, Road_Type, Road_Condition, Weather,
            Traffic_Volume, Average_Speed_kmph, Congestion_Level, Accident_Occurred, etc.</li>
        <li><b>⚠️ Synthetic data is clearly labelled as illustrative</b> — it does not
            represent real-world traffic measurements.</li>
    </ul>
    </div>

    <div class="insight-box">
    <b>⚙️ Methodology</b>
    <ul>
        <li><b>Cleaning:</b> Deduplication, median/mode imputation, outlier clipping.</li>
        <li><b>Feature engineering:</b> Peak-hour flag, time-period label, accident rate.</li>
        <li><b>EDA:</b> Distribution analysis, temporal trends, correlation heatmaps.</li>
        <li><b>ML:</b> Random Forest / Gradient Boosting / Logistic Regression via
            scikit-learn Pipeline (Imputer → Scaler → Classifier).</li>
        <li><b>Metrics:</b> Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.</li>
    </ul>
    </div>

    <div class="insight-box">
    <b>🛠️ Technologies</b>
    <ul>
        <li>Python 3.10+ · Streamlit · Pandas · NumPy · Plotly · Matplotlib ·
            Seaborn · Scikit-learn</li>
    </ul>
    </div>

    <div class="insight-box">
    <b>⚠️ Limitations</b>
    <ul>
        <li>Synthetic data uses simplified distributions — real traffic is more complex.</li>
        <li>No real-time streaming support in this version.</li>
        <li>Map view requires valid latitude/longitude columns (synthetic data includes them).</li>
        <li>Class imbalance in accident prediction not corrected (SMOTE not applied).</li>
    </ul>
    </div>

    <div class="insight-box">
    <b>🚀 Future Enhancements</b>
    <ul>
        <li>Real-time data integration via traffic APIs (TomTom, HERE).</li>
        <li>LSTM/Transformer time-series models for congestion forecasting.</li>
        <li>Route optimisation module.</li>
        <li>Cloud deployment (Streamlit Community Cloud, AWS, GCP).</li>
    </ul>
    </div>

    <div class="insight-box">
    <b>👤 Author:</b> Miroslav Mandi &nbsp;|&nbsp;
    <b>Version:</b> 1.0 &nbsp;|&nbsp; <b>Year:</b> 2026
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Banner ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b4b,#1a2d6b,#0d1b4b);
                border-radius:14px;padding:24px 32px;margin-bottom:16px;
                border:1px solid #2e4070;">
        <h1 style="color:#e6f1ff;margin:0;font-size:2rem;font-weight:800;">
            🚦 TrafficPulse AI
        </h1>
        <p style="color:#8892b0;margin:6px 0 0;font-size:15px;">
            Traffic, Congestion &amp; Accident Analytics Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Data source ───────────────────────────────────────────────────────
    data_source = st.radio(
        "Data Source",
        ["📂 Upload CSV Dataset", "🤖 Use Synthetic Dataset"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_df = None

    if data_source == "📂 Upload CSV Dataset":
        st.markdown(
            '<div class="upload-box">📁 Upload your traffic CSV file</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                    label_visibility="collapsed")
        if uploaded:
            try:
                raw_df = load_csv(uploaded.read(), uploaded.name)
                st.success(f"✅ Loaded **{uploaded.name}** — "
                           f"{len(raw_df):,} rows × {raw_df.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
        else:
            st.info("👆 Upload a CSV, or switch to Synthetic Dataset mode.")
    else:
        n_rows = st.select_slider(
            "Synthetic Data Size",
            options=[1000, 2000, 5000, 10000],
            value=5000,
        )
        with st.spinner("Generating synthetic dataset…"):
            raw_df = generate_synthetic_data(n=n_rows)
        st.success(f"🤖 Synthetic dataset generated — {len(raw_df):,} rows")
        st.caption(
            "⚠️ **Synthetic / Illustrative Data** — "
            "not real-world measurements. All statistics are generated."
        )

    if raw_df is None:
        st.stop()

    # ── Pre-process ───────────────────────────────────────────────────────
    with st.spinner("Cleaning and preparing data…"):
        df = clean_data(map_columns(raw_df))

    # ── Sidebar + filters ─────────────────────────────────────────────────
    page, date_range, sel_loc, sel_weather, sel_rtype, sel_rcond, sel_day, sel_cong = \
        build_sidebar(df)

    fdf = apply_filters(df, date_range, sel_loc, sel_weather,
                        sel_rtype, sel_rcond, sel_day, sel_cong)

    # ── Route ─────────────────────────────────────────────────────────────
    if   page == "🏠 Overview":           page_overview(fdf)
    elif page == "📈 Traffic Analysis":   page_traffic(fdf)
    elif page == "⚠️ Accident Analysis":  page_accident(fdf)
    elif page == "🚧 Congestion Analysis":page_congestion(fdf)
    elif page == "🗺️ Map View":           page_map(fdf)
    elif page == "🤖 ML Predictions":     page_ml(fdf)
    elif page == "ℹ️ About Project":      page_about()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">TrafficPulse AI — Miroslav Mandi © 2024 | '
        'Streamlit · Plotly · Scikit-learn</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
