"""VoltSentinel — AI-powered energy monitoring dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Support `streamlit run app.py` from a plain checkout, without an install step.
SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voltsentinel import __version__, charts, metrics, theme  # noqa: E402
from voltsentinel.config import Settings, settings  # noqa: E402
from voltsentinel.data import DataValidationError, load_dataset  # noqa: E402
from voltsentinel.detection import DETECTOR_LABELS, detect  # noqa: E402

st.set_page_config(
    page_title="VoltSentinel · Energy Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.register()


# --------------------------------------------------------------------------
# Cached pipeline
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading telemetry…")
def _load(upload: bytes | None) -> pd.DataFrame:
    """Load the bundled dataset, or an uploaded CSV when one is supplied."""
    if upload is None:
        return load_dataset()
    from io import BytesIO

    return load_dataset(BytesIO(upload))


@st.cache_data(show_spinner="Scoring readings…")
def _detect(df: pd.DataFrame, tuning: tuple) -> pd.DataFrame:
    """Run detection. ``tuning`` is part of the cache key, not the payload."""
    (contamination, zscore_threshold, phantom_load, voltage_min, voltage_max) = tuning
    config = Settings(
        data_path=settings.data_path,
        contamination=contamination,
        n_estimators=settings.n_estimators,
        random_state=settings.random_state,
        zscore_threshold=zscore_threshold,
        phantom_load_kwh=phantom_load,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
    )
    return detect(df, config)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("⚡ VoltSentinel")
st.sidebar.caption(f"v{__version__}")

uploaded = st.sidebar.file_uploader(
    "Telemetry CSV",
    type="csv",
    help="Leave empty to use the bundled sample dataset.",
)

try:
    data = _load(uploaded.getvalue() if uploaded else None)
except DataValidationError as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

with st.sidebar.expander("Detection settings", expanded=False):
    contamination = st.slider(
        "Expected anomaly rate",
        min_value=0.005,
        max_value=0.15,
        value=settings.contamination,
        step=0.005,
        format="%.3f",
        help="Share of readings the Isolation Forest may flag.",
    )
    zscore_threshold = st.slider(
        "Outlier sensitivity (robust z)",
        min_value=1.0,
        max_value=6.0,
        value=settings.zscore_threshold,
        step=0.1,
        help="Lower is stricter. Computed per device type.",
    )
    phantom_load = st.slider(
        "Phantom-load threshold (kWh)",
        min_value=10.0,
        max_value=150.0,
        value=settings.phantom_load_kwh,
        step=5.0,
        help="Draw above this while a device reports OFF is an anomaly.",
    )
    voltage_min, voltage_max = st.slider(
        "Acceptable voltage band (V)",
        min_value=200.0,
        max_value=260.0,
        value=(settings.voltage_min, settings.voltage_max),
        step=0.5,
    )

scored = _detect(
    data,
    (contamination, zscore_threshold, phantom_load, voltage_min, voltage_max),
)

st.sidebar.header("Filters")
device_types = sorted(scored["device_type"].unique())
selected_types = st.sidebar.multiselect("Device type", options=device_types, default=device_types)

device_ids = sorted(scored[scored["device_type"].isin(selected_types)]["device_id"].unique())
selected_ids = st.sidebar.multiselect(
    "Device", options=device_ids, default=device_ids, help="Defaults to every device."
)

min_date: date = scored["timestamp"].min().date()
max_date: date = scored["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:  # the user is mid-selection and has only picked one end
    start_date = end_date = date_range if isinstance(date_range, date) else min_date

statuses = sorted(scored["status"].unique())
selected_statuses = st.sidebar.multiselect("Status", options=statuses, default=statuses)
anomalies_only = st.sidebar.toggle("Anomalies only", value=False)

# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
mask = (
    scored["device_type"].isin(selected_types)
    & scored["device_id"].isin(selected_ids)
    & scored["status"].isin(selected_statuses)
    & (scored["timestamp"].dt.date >= start_date)
    & (scored["timestamp"].dt.date <= end_date)
)
view = scored[mask]
if anomalies_only:
    view = view[view["detected_anomaly"] == 1]

st.title("⚡ VoltSentinel")
st.caption(
    "Agentic anomaly detection across the device fleet — Isolation Forest, "
    "robust per-type outlier scoring and engineering rules, run over every reading."
)

if view.empty:
    st.warning("No readings match the current filters. Widen the selection in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------
# Headline metrics
# --------------------------------------------------------------------------
summary = metrics.summarise(view)
baseline_summary = metrics.summarise(scored)

kpi = st.columns(5)
kpi[0].metric(
    "Total usage",
    f"{summary.total_usage_kwh:,.0f} kWh",
    delta=f"{summary.total_usage_kwh - baseline_summary.total_usage_kwh:,.0f} vs. fleet",
    delta_color="off",
)
kpi[1].metric("Peak load", f"{summary.peak_load_kwh:,.1f} kWh")
kpi[2].metric("Avg. voltage", f"{summary.average_voltage:,.1f} V")
kpi[3].metric(
    "Anomalies",
    f"{summary.anomaly_count:,}",
    delta=f"{summary.anomaly_rate:.1f}% of readings",
    delta_color="inverse",
)
kpi[4].metric("Devices online", f"{summary.uptime_pct:.0f}%")

st.caption(
    f"{summary.reading_count:,} readings · {summary.device_count} devices · "
    f"busiest type: {summary.busiest_device_type}"
)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
overview_tab, anomaly_tab, devices_tab, data_tab = st.tabs(
    ["Overview", "Anomalies", "Devices", "Data"]
)

with overview_tab:
    flagged = view[view["detected_anomaly"] == 1]

    # A week of hourly readings across five device types is a hairball, so let
    # the reader roll the timeline up. The default follows the span selected.
    span_days = (view["timestamp"].max() - view["timestamp"].min()).days
    granularities = {"Hourly": "h", "6-hourly": "6h", "Daily": "D"}
    default_granularity = "Hourly" if span_days <= 2 else "6-hourly"
    granularity = st.segmented_control(
        "Granularity",
        options=list(granularities),
        default=default_granularity,
        label_visibility="collapsed",
    )
    frequency = granularities.get(granularity or default_granularity, "h")

    st.plotly_chart(
        charts.usage_timeline(metrics.usage_over_time(view, frequency), flagged),
        width="stretch",
    )
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.usage_by_type(metrics.usage_by_device_type(view)), width="stretch")
    with right:
        st.plotly_chart(charts.hourly_heatmap(metrics.hourly_profile(view)), width="stretch")
    st.plotly_chart(charts.voltage_band(view, voltage_min, voltage_max), width="stretch")

with anomaly_tab:
    flagged = view[view["detected_anomaly"] == 1]
    if flagged.empty:
        st.success("No anomalies detected in the current selection.")
    else:
        breakdown = st.columns(len(DETECTOR_LABELS))
        for column, (flag, label) in zip(breakdown, DETECTOR_LABELS.items(), strict=False):
            column.metric(label, int(view[flag].sum()))

        st.plotly_chart(charts.anomaly_scatter(view), width="stretch")

        st.subheader("Flagged readings")
        table = metrics.anomaly_table(view)
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="MMM D, HH:mm"),
                "device_id": st.column_config.TextColumn("Device"),
                "device_type": st.column_config.TextColumn("Type"),
                "status": st.column_config.TextColumn("Status"),
                "usage_kwh": st.column_config.NumberColumn("kWh", format="%.2f"),
                "voltage": st.column_config.NumberColumn("Voltage", format="%.2f V"),
                "anomaly_score": st.column_config.ProgressColumn(
                    "Score", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "anomaly_reasons": st.column_config.TextColumn("Why it was flagged", width="large"),
            },
        )
        st.download_button(
            "Download anomalies (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="voltsentinel_anomalies.csv",
            mime="text/csv",
            icon=":material/download:",
        )

        if "baseline_ai_flag" in view.columns:
            agreement = int(
                ((view["baseline_ai_flag"] == 1) & (view["detected_anomaly"] == 1)).sum()
            )
            st.caption(
                f"Baseline flags shipped with the dataset: {int(view['baseline_ai_flag'].sum())} "
                f"· agreeing with the live model: {agreement}"
            )

with devices_tab:
    leaderboard = metrics.device_leaderboard(view)
    st.subheader("Devices ranked by anomalies raised")
    st.dataframe(
        leaderboard,
        width="stretch",
        hide_index=True,
        column_config={
            "device_id": st.column_config.TextColumn("Device"),
            "device_type": st.column_config.TextColumn("Type"),
            "total_kwh": st.column_config.NumberColumn("Total kWh", format="%.1f"),
            "mean_voltage": st.column_config.NumberColumn("Avg. voltage", format="%.1f V"),
            "anomalies": st.column_config.NumberColumn("Anomalies", format="%d"),
            "anomaly_rate_pct": st.column_config.NumberColumn("Anomaly rate", format="%.1f%%"),
        },
    )
    st.subheader("Consumption by device type")
    st.dataframe(
        metrics.usage_by_device_type(view),
        width="stretch",
        hide_index=True,
        column_config={
            "device_type": st.column_config.TextColumn("Type"),
            "total_kwh": st.column_config.NumberColumn("Total kWh", format="%.1f"),
            "mean_kwh": st.column_config.NumberColumn("Mean kWh", format="%.2f"),
            "anomalies": st.column_config.NumberColumn("Anomalies", format="%d"),
            "readings": st.column_config.NumberColumn("Readings", format="%d"),
        },
    )

with data_tab:
    st.subheader("Filtered readings")
    st.dataframe(view, width="stretch", hide_index=True, height=460)
    st.download_button(
        "Download selection (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="voltsentinel_selection.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    with st.expander("How readings are scored"):
        st.markdown(
            "\n".join(f"- **{label}** (`{flag}`)" for flag, label in DETECTOR_LABELS.items())
        )
        st.markdown(
            "`anomaly_score` blends the Isolation Forest score (60%) with how many "
            "detectors agree (40%), so a reading that trips several checks outranks "
            "one that only just crossed a threshold."
        )

st.divider()
st.caption("Developed by ArtifexLabs · Agentic AI for energy monitoring")
