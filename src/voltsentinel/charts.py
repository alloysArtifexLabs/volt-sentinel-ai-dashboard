"""Plotly figures for the dashboard.

Every figure returns a ``go.Figure`` and takes already-aggregated data, which
keeps the charting free of business logic and easy to test.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from voltsentinel import theme

MARKER_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "hexagon"]


def _empty(message: str) -> go.Figure:
    """A placeholder figure so an empty selection never renders a broken axis."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font={"color": theme.INK_MUTED, "size": 13},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    figure.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, height=280)
    return figure


def usage_timeline(usage: pd.DataFrame, anomalies: pd.DataFrame | None = None) -> go.Figure:
    """Load over time per device type, with flagged readings marked on top."""
    if usage.empty:
        return _empty("No readings match the current filters.")

    colors = theme.color_map(usage["device_type"].unique().tolist())
    figure = px.line(
        usage,
        x="timestamp",
        y="usage_kwh",
        color="device_type",
        color_discrete_map=colors,
        title="Average load over time",
        labels={
            "usage_kwh": "kWh per reading",
            "timestamp": "",
            "device_type": "Device type",
        },
    )
    figure.update_traces(
        line={"width": 2}, hovertemplate="%{y:.1f} kWh<extra>%{fullData.name}</extra>"
    )

    if anomalies is not None and not anomalies.empty:
        figure.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["usage_kwh"],
                mode="markers",
                name="Anomaly",
                marker={
                    "size": 10,
                    "color": theme.STATUS["critical"],
                    "symbol": "circle-open",
                    # A surface-coloured ring keeps overlapping marks readable.
                    "line": {"width": 2, "color": theme.STATUS["critical"]},
                },
                customdata=anomalies[["device_id", "anomaly_reasons"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{y:.1f} kWh<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )

    figure.update_layout(hovermode="x unified", height=420, legend_title_text="")
    return figure


def usage_by_type(summary: pd.DataFrame) -> go.Figure:
    """Total consumption per device type — one measure, so one colour."""
    if summary.empty:
        return _empty("No readings match the current filters.")

    figure = go.Figure(
        go.Bar(
            x=summary["device_type"],
            y=summary["total_kwh"],
            marker={"color": theme.CATEGORICAL[0], "cornerradius": 4},
            text=summary["total_kwh"].round(0),
            texttemplate="%{text:,.0f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:,.1f} kWh<extra></extra>",
        )
    )
    figure.update_layout(
        title="Total consumption by device type",
        # Bars encode magnitude by length, so the axis must reach zero.
        yaxis={"title": "kWh", "rangemode": "tozero"},
        bargap=0.35,
        height=380,
        showlegend=False,
    )
    return figure


def hourly_heatmap(profile: pd.DataFrame) -> go.Figure:
    """Average load for every (device type, hour) cell — magnitude, so one ramp."""
    if profile.empty:
        return _empty("No readings match the current filters.")

    grid = profile.pivot(index="device_type", columns="hour", values="usage_kwh")
    figure = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[f"{hour:02d}" for hour in grid.columns],
            y=grid.index.tolist(),
            colorscale=[[i / 6, c] for i, c in enumerate(theme.SEQUENTIAL_BLUE)],
            xgap=2,
            ygap=2,
            colorbar={
                "title": {"text": "kWh", "font": {"size": 11, "color": theme.INK_SECONDARY}},
                "thickness": 12,
                "outlinewidth": 0,
            },
            hovertemplate="<b>%{y}</b><br>%{x}:00 · %{z:.1f} kWh avg<extra></extra>",
        )
    )
    figure.update_layout(
        title="Average load by hour of day",
        xaxis={"title": "Hour", "showgrid": False},
        yaxis={"showgrid": False},
        height=340,
    )
    return figure


def anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    """Usage against voltage, shaded by anomaly score.

    Device identity rides on the marker symbol rather than a sixth hue, which
    keeps the pairwise colour separation safe for colour-vision deficiency.
    """
    if df.empty:
        return _empty("No readings match the current filters.")

    device_types = sorted(df["device_type"].unique())
    symbols = {
        device: MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)]
        for index, device in enumerate(device_types)
    }

    figure = px.scatter(
        df,
        x="voltage",
        y="usage_kwh",
        color="anomaly_score",
        symbol="device_type",
        symbol_map=symbols,
        # Skip the palest step: on a white surface it would render the
        # low-score marks nearly invisible.
        color_continuous_scale=theme.SEQUENTIAL_BLUE[1:],
        range_color=(0, 1),
        title="Load vs. supply voltage, shaded by anomaly score",
        labels={
            "voltage": "Voltage (V)",
            "usage_kwh": "kWh",
            "anomaly_score": "Score",
            "device_type": "Device type",
        },
        hover_data={"device_id": True, "anomaly_reasons": True, "anomaly_score": ":.2f"},
    )
    figure.update_traces(
        marker={"size": 9, "opacity": 0.85, "line": {"width": 1, "color": theme.SURFACE}}
    )
    figure.update_layout(height=440, legend_title_text="")
    return figure


def voltage_band(df: pd.DataFrame, low: float, high: float) -> go.Figure:
    """Voltage over time with the acceptable band shaded behind the traces."""
    if df.empty:
        return _empty("No readings match the current filters.")

    colors = theme.color_map(df["device_type"].unique().tolist())
    series = df.groupby(["timestamp", "device_type"])["voltage"].mean().reset_index()
    figure = px.line(
        series,
        x="timestamp",
        y="voltage",
        color="device_type",
        color_discrete_map=colors,
        title="Supply voltage against the acceptable band",
        labels={"voltage": "Volts", "timestamp": "", "device_type": "Device type"},
    )
    figure.update_traces(
        line={"width": 2}, hovertemplate="%{y:.1f} V<extra>%{fullData.name}</extra>"
    )
    figure.add_hrect(
        y0=low,
        y1=high,
        fillcolor=theme.STATUS["good"],
        opacity=0.06,
        line_width=0,
        layer="below",
    )
    for bound, label in ((low, f"{low:g} V min"), (high, f"{high:g} V max")):
        figure.add_hline(
            y=bound,
            line={"width": 1, "dash": "dot", "color": theme.INK_MUTED},
            annotation_text=label,
            annotation_position="right",
            annotation_font={"size": 11, "color": theme.INK_MUTED},
        )
    figure.update_layout(hovermode="x unified", height=380, legend_title_text="")
    return figure
