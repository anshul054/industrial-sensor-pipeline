"""
streamlit_app.py
------------------
Interactive demo for the industrial sensor ETL/feature pipeline.
Lets a visitor upload their own sensor CSV (or use the built-in sample
data), runs it through ingest.py -> transform.py, and visualizes the
resampled readings, rolling stats, and flagged anomalies per machine.

Run locally:
    streamlit run streamlit_app.py

Deploy: push this repo to GitHub, then deploy for free on
Streamlit Community Cloud (share.streamlit.io), pointing it at this file.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ingest import load_raw_readings  # noqa: E402
from transform import run_transform, SENSOR_COLS  # noqa: E402

st.set_page_config(page_title="Industrial Sensor Pipeline", layout="wide")

st.title("Industrial Sensor ETL / Feature Pipeline")
st.markdown(
    "Upload multi-machine sensor data (or use the sample data below) to see it "
    "cleaned, resampled onto a 5-min grid, feature-engineered, and scanned for "
    "anomalies with a per-machine rolling z-score."
)

REQUIRED_COLS = ["timestamp", "machine_id", "temperature_c", "vibration_mm_s", "pressure_psi", "rpm"]


@st.cache_data
def generate_sample_data(machines: int = 5, days: int = 3, seed: int = 42) -> pd.DataFrame:
    """Small synthetic dataset so the demo works with zero setup."""
    rng = np.random.default_rng(seed)
    periods = days * 24 * 60  # 1-min steps
    start = pd.Timestamp("2026-06-01")
    frames = []
    for i in range(1, machines + 1):
        machine_id = f"M-{i:03d}"
        t = np.arange(periods)
        ts = pd.date_range(start=start, periods=periods, freq="1min")
        temperature = 65 + 4 * np.sin(t / 180) + rng.normal(0, 0.8, periods)
        vibration = 2.0 + 0.3 * np.sin(t / 45 + 1) + rng.normal(0, 0.15, periods)
        pressure = 100 + 5 * np.sin(t / 300 + 2) + rng.normal(0, 1.2, periods)
        rpm = 1500 + 50 * np.sin(t / 90) + rng.normal(0, 8, periods)

        df = pd.DataFrame({
            "timestamp": ts, "machine_id": machine_id,
            "temperature_c": temperature, "vibration_mm_s": vibration,
            "pressure_psi": pressure, "rpm": rpm,
        })

        # inject a few overheat spikes and a stall, so the demo has something to find
        spike_idx = rng.choice(periods, size=max(1, periods // 600), replace=False)
        df.loc[spike_idx, "temperature_c"] += rng.uniform(20, 40, len(spike_idx))
        df.loc[spike_idx, "vibration_mm_s"] += rng.uniform(3, 6, len(spike_idx))

        stall_idx = rng.choice(periods, size=max(1, periods // 900), replace=False)
        df.loc[stall_idx, "rpm"] = 0

        # sprinkle some missing values and duplicate rows, like the real pipeline expects
        miss_idx = rng.choice(periods, size=int(periods * 0.01), replace=False)
        df.loc[miss_idx, "temperature_c"] = np.nan
        dup_rows = df.sample(frac=0.003, random_state=int(rng.integers(0, 1_000_000)))
        df = pd.concat([df, dup_rows], ignore_index=True)

        frames.append(df)

    return pd.concat(frames, ignore_index=True).sort_values(["machine_id", "timestamp"]).reset_index(drop=True)


with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    st.caption(f"Required columns: {', '.join(REQUIRED_COLS)}")
    st.divider()
    st.header("About")
    st.markdown(
        "This dashboard is a live demo of a Python ETL/feature pipeline built for "
        "multi-machine industrial sensor data. [View source on GitHub]"
        "(https://github.com/anshul054/industrial-sensor-pipeline)"
    )

if uploaded is not None:
    raw_path = "uploaded_temp.csv"
    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())
    try:
        raw_df = load_raw_readings(raw_path)
    except Exception as e:
        st.error(f"Could not load file: {e}")
        st.stop()
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)
    st.success(f"Loaded {len(raw_df):,} rows from your upload.")
else:
    raw_df = generate_sample_data()
    st.info(f"Using built-in sample data: {len(raw_df):,} rows across {raw_df['machine_id'].nunique()} machines. Upload your own CSV in the sidebar to try it on real data.")

with st.spinner("Running pipeline: dedupe -> resample -> features -> anomaly detection..."):
    result = run_transform(raw_df)

machines = sorted(result["machine_id"].unique())
col1, col2, col3, col4 = st.columns(4)
col1.metric("Machines", len(machines))
col2.metric("Rows (post-resample)", f"{len(result):,}")
col3.metric("Anomalies flagged", f"{int(result['any_anomaly'].sum()):,}")
anomaly_rate = 100 * result["any_anomaly"].sum() / len(result) if len(result) else 0
col4.metric("Anomaly rate", f"{anomaly_rate:.2f}%")

st.divider()

selected_machine = st.selectbox("Select a machine to inspect", machines)
sensor = st.selectbox("Select a sensor", SENSOR_COLS, format_func=lambda c: c.replace("_", " ").title())

machine_df = result[result["machine_id"] == selected_machine].sort_values("timestamp")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=machine_df["timestamp"], y=machine_df[sensor],
    mode="lines", name=sensor, line=dict(color="#4C9AFF"),
))
fig.add_trace(go.Scatter(
    x=machine_df["timestamp"], y=machine_df[f"{sensor}_roll_mean"],
    mode="lines", name="rolling mean", line=dict(color="#999999", dash="dash"),
))
anomalies = machine_df[machine_df[f"{sensor}_anomaly"]]
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"], y=anomalies[sensor],
        mode="markers", name="flagged anomaly",
        marker=dict(color="#E5484D", size=9, symbol="x"),
    ))
fig.update_layout(
    title=f"{sensor.replace('_', ' ').title()} — {selected_machine}",
    xaxis_title="Time", yaxis_title=sensor, height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Flagged anomalies (this machine)")
anomaly_cols = ["timestamp", "machine_id"] + SENSOR_COLS + ["any_anomaly", "hours_since_maintenance"]
st.dataframe(
    machine_df[machine_df["any_anomaly"]][anomaly_cols].reset_index(drop=True),
    use_container_width=True,
)

with st.expander("View full feature table for this machine"):
    st.dataframe(machine_df.reset_index(drop=True), use_container_width=True)

st.divider()
st.caption(
    "Pipeline: drop duplicates -> resample to 5-min grid per machine -> forward-fill gaps ≤15min -> "
    "1-hour rolling mean/std per sensor -> per-machine rolling z-score anomaly flags -> simulated "
    "maintenance feature. Source: ingest.py + transform.py in this repo."
)
