import os
import calendar
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import shap


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bangladesh Air Quality Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

def load_css():

    css_candidates = [
        os.path.join(os.path.dirname(__file__), "css", "style.css"),
        os.path.join(os.path.dirname(__file__), "style.css")
    ]
    css_path = next((path for path in css_candidates if os.path.exists(path)), None)

    if css_path:

        with open(
            css_path,
            "r",
            encoding="utf-8"
        ) as file:

            css = file.read()

        st.html(
            f"<style>{css}</style>"
        )


load_css()


# ============================================================
# DIRECT DASHBOARD MODE
# ============================================================
# Standalone research/demo build: no accounts and no MySQL dependency.

# ============================================================
# LOAD CSV DATA
# ============================================================

csv_path = os.path.join(
    os.path.dirname(__file__),
    "air_quality_data.csv"
)


try:

    data = pd.read_csv(
        csv_path
    )

except FileNotFoundError:

    st.error(
        "air_quality_data.csv was not found. "
        "Please place it in the same folder as app.py."
    )

    st.stop()


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

required_columns = [

    "year",

    "month",

    "CO_mean",

    "NO2_mean",

    "SO2_mean",

    "O3_mean",

    "Temp_mean",

    "Humidity_mean"

]


missing_columns = [

    column

    for column in required_columns

    if column not in data.columns

]


if missing_columns:

    st.error(
        f"Missing columns in CSV: {missing_columns}"
    )

    st.stop()


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.html(
    """
    <div class="dashboard-header">

        <div class="dashboard-title">
            🌍 Bangladesh Air Quality
        </div>

        <div class="dashboard-subtitle">
            Monitoring & Machine Learning Dashboard
        </div>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("Dashboard Controls")

    pollutant = st.selectbox("Select Pollutant", ["CO", "NO2", "SO2", "O3"])

    years = sorted(data["year"].dropna().unique())

    year = st.selectbox("Select Year", years)


# ============================================================
# FILTER DATA
# ============================================================

filtered_data = data[
    data["year"] == year
].copy()


# ============================================================
# POLLUTANT COLUMN MAP
# ============================================================

pollutant_column_map = {

    "CO": "CO_mean",

    "NO2": "NO2_mean",

    "SO2": "SO2_mean",

    "O3": "O3_mean"

}


pollutant_col = pollutant_column_map[
    pollutant
]


# ============================================================
# METRIC DATA
# ============================================================

pollution_values = filtered_data[
    pollutant_col
].dropna()


temperature_values = filtered_data[
    "Temp_mean"
].dropna()


humidity_values = filtered_data[
    "Humidity_mean"
].dropna()


# ============================================================
# METRIC CARDS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Selected Pollutant",
        pollutant
    )


with metric2:

    if len(pollution_values) > 0:

        st.metric(
            f"Average {pollutant}",
            f"{pollution_values.mean():.4f}"
        )

    else:

        st.metric(
            f"Average {pollutant}",
            "N/A"
        )


with metric3:

    if len(temperature_values) > 0:

        st.metric(
            "Average Temperature",
            f"{temperature_values.mean() - 273.15:.2f} °C"
        )

    else:

        st.metric(
            "Average Temperature",
            "N/A"
        )


with metric4:

    if len(humidity_values) > 0:

        st.metric(
            "Average Humidity",
            f"{humidity_values.mean():.2f} %"
        )

    else:

        st.metric(
            "Average Humidity",
            "N/A"
        )


st.write("")


# ============================================================
# TOP ROW
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# MONTHLY TREND
# ============================================================

with col1:

    st.subheader(
        f"📈 {pollutant} Monthly Trend - {year}"
    )

    plot_data = filtered_data.dropna(
        subset=[
            "month",
            pollutant_col
        ]
    ).sort_values(
        "month"
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        plot_data["month"],
        plot_data[pollutant_col],
        marker="o",
        linewidth=2.5,
        color="#2563eb"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        f"{pollutant} Concentration"
    )

    if pd.api.types.is_numeric_dtype(plot_data["month"]):
        # One tick per unique month, shown as Jan, Feb, ... instead of raw numbers
        month_ticks = sorted(plot_data["month"].unique())
        month_labels = [
            calendar.month_abbr[int(m)] if 1 <= int(m) <= 12 else str(m)
            for m in month_ticks
        ]
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, rotation=45, ha="right")
    else:
        # Month names stored as text: just rotate so they don't collide
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    st.pyplot(fig)


# ============================================================
# MONTHLY TABLE
# ============================================================

with col2:

    st.subheader(
        "📋 Monthly Data"
    )

    st.dataframe(
        filtered_data[
            [
                "month",
                pollutant_col,
                "Temp_mean",
                "Humidity_mean"
            ]
        ],
        height=330,
        use_container_width=True
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

st.divider()
st.subheader("🤖 CatBoost & Random Forest Prediction + XAI")
st.caption(
    "CatBoost is used for temperature prediction and Random Forest is used for humidity prediction. "
    "SHAP is used to explain model predictions; it does not establish causation."
)

model_features = ["CO_mean", "NO2_mean", "SO2_mean", "O3_mean"]
feature_labels = ["CO", "NO₂", "SO₂", "O₃"]

# CatBoost parameters (temperature)
catboost_params = dict(
    loss_function="RMSE",
    iterations=300,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
    allow_writing_files=False   # avoids creating a catboost_info/ folder
)

# Random Forest parameters (humidity)
rf_params = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

temp_data = data.dropna(subset=model_features + ["Temp_mean"])
hum_data = data.dropna(subset=model_features + ["Humidity_mean"])


@st.cache_resource(show_spinner="Training CatBoost temperature model...")
def train_temperature_model(X, y):
    model = CatBoostRegressor(**catboost_params)
    model.fit(X, y)
    return model


@st.cache_resource(show_spinner="Training Random Forest humidity model...")
def train_humidity_model(X, y):
    model = RandomForestRegressor(**rf_params)
    model.fit(X, y)
    return model


model_temp = train_temperature_model(temp_data[model_features], temp_data["Temp_mean"])
model_hum = train_humidity_model(hum_data[model_features], hum_data["Humidity_mean"])

st.subheader("Pollution Scenario Inputs")
st.caption("Values are constrained to the observed ranges in the research dataset.")

input_cols = st.columns(4)
ranges = {feature: (float(data[feature].min()), float(data[feature].max())) for feature in model_features}
input_values = {}

for col, feature, label in zip(input_cols, model_features, feature_labels):
    lo, hi = ranges[feature]
    default = float(data[feature].median())
    with col:
        if hi <= lo:
            input_values[feature] = lo
            st.number_input(label, value=lo, disabled=True, key=f"ml_{feature}")
        else:
            step = max((hi - lo) / 100.0, 1e-12)
            fmt = "%.6f" if hi >= 0.001 else "%.8f"
            input_values[feature] = st.slider(label, min_value=lo, max_value=hi, value=default, step=step, format=fmt, key=f"ml_{feature}")

input_data = pd.DataFrame([[input_values[f] for f in model_features]], columns=model_features)

pred_temp_kelvin = float(model_temp.predict(input_data)[0])
pred_temp_celsius = pred_temp_kelvin - 273.15
pred_humidity = float(model_hum.predict(input_data)[0])

prediction1, prediction2 = st.columns(2)
with prediction1:
    st.html(f'<div class="prediction-card temperature-card"><div class="prediction-label">Model-Estimated Temperature (CatBoost)</div><div class="prediction-value">{pred_temp_celsius:.2f} °C</div></div>')
with prediction2:
    st.html(f'<div class="prediction-card humidity-card"><div class="prediction-label">Model-Estimated Humidity (Random Forest)</div><div class="prediction-value">{pred_humidity:.2f} %</div></div>')

st.divider()
st.subheader("🔍 XAI — SHAP Predictive Contribution")
st.caption("Global SHAP importance summarizes average absolute contribution across the observations used by each model. Local SHAP values show how the current scenario contributes to each prediction.")


@st.cache_data(show_spinner="Computing global SHAP values...")
def compute_global_shap(model_name, _model, X, labels):
    # model_name is part of the cache key, so each model gets its own cached result
    explainer = shap.TreeExplainer(_model)
    values = explainer.shap_values(X, check_additivity=False)
    series = pd.Series(abs(values).mean(axis=0), index=labels)
    if series.sum():
        series = series / series.sum()
    return series


shap_temp = shap.TreeExplainer(model_temp)
shap_hum = shap.TreeExplainer(model_hum)

global_temp = compute_global_shap("catboost_temperature", model_temp, temp_data[model_features], feature_labels)
global_hum = compute_global_shap("random_forest_humidity", model_hum, hum_data[model_features], feature_labels)

local_temp = shap_temp.shap_values(input_data, check_additivity=False)[0]
local_hum = shap_hum.shap_values(input_data, check_additivity=False)[0]

xai_col1, xai_col2 = st.columns(2)
for target_name, global_values, local_values, parent_col in [
    ("Temperature (CatBoost)", global_temp, local_temp, xai_col1),
    ("Humidity (Random Forest)", global_hum, local_hum, xai_col2),
]:
    with parent_col:
        st.markdown(f"### {target_name}")
        fig_xai, ax_xai = plt.subplots(figsize=(6.6, 3.8), facecolor="#edf4fb")
        ax_xai.set_facecolor("#e7f1f7")
        bars = ax_xai.barh(list(global_values.index), list(global_values.values), color="#3158b7")
        ax_xai.invert_yaxis()
        ax_xai.set_xlabel("Mean |SHAP value| (normalized)", color="#17324d")
        ax_xai.tick_params(colors="#17324d")
        ax_xai.grid(axis="x", alpha=0.20)
        for spine in ax_xai.spines.values():
            spine.set_color("#c7d9df")
        max_x = max(global_values.values) if len(global_values) else 0
        for bar in bars:
            ax_xai.text(bar.get_width() + (max_x * 0.02 if max_x else 0.01), bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}", va="center", color="#17324d")
        fig_xai.tight_layout()
        st.pyplot(fig_xai)
        local_df = pd.DataFrame({"Pollutant": feature_labels, "SHAP contribution": local_values})
        st.dataframe(local_df, hide_index=True, use_container_width=True)

st.markdown('<div class="xai-card"><strong>Interpretation:</strong> SHAP values explain the contribution of each pollutant to the fitted model output. They are predictive explanations, not evidence of a causal atmospheric effect.</div>', unsafe_allow_html=True)

st.divider()
st.subheader("🧪 Scenario Sensitivity")
st.caption("Each pollutant is varied across its observed range while the other current inputs are held fixed. The resulting changes are model-estimated scenario responses.")

sens_rows = []
for feature, label in zip(model_features, feature_labels):
    lo, hi = ranges[feature]
    low_case = input_data.copy(); high_case = input_data.copy()
    low_case[feature] = lo; high_case[feature] = hi
    t_low = float(model_temp.predict(low_case)[0]) - 273.15
    t_high = float(model_temp.predict(high_case)[0]) - 273.15
    h_low = float(model_hum.predict(low_case)[0])
    h_high = float(model_hum.predict(high_case)[0])
    sens_rows.append({
        "Pollutant": label,
        "Temperature change (°C)": t_high - t_low,
        "Humidity change (percentage points)": h_high - h_low
    })

st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)
st.caption("Scenario outputs are model-based sensitivity results and should not be interpreted as causal effects.")

# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
  <div>Bangladesh Air Quality Monitoring Dashboard</div>
  <div>Data: 2020–2025 | Models: CatBoost (Temperature), Random Forest (Humidity) | XAI: SHAP | Mode: Standalone Research Dashboard</div>
</div>
""")