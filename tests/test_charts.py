from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from voltsentinel import charts, metrics, theme
from voltsentinel.detection import detect


def test_theme_registers_as_the_plotly_default() -> None:
    import plotly.io as pio

    assert theme.register() == theme.TEMPLATE_NAME
    assert pio.templates.default == theme.TEMPLATE_NAME


def test_colour_follows_the_entity_not_the_selection() -> None:
    full = theme.color_map(["HVAC", "Lighting", "Pump"])
    filtered = theme.color_map(["HVAC", "Lighting", "Pump"])
    assert full == filtered
    assert len(set(full.values())) == 3


def test_every_chart_renders(frame: pd.DataFrame) -> None:
    result = detect(frame)
    flagged = result[result["detected_anomaly"] == 1]
    figures = [
        charts.usage_timeline(metrics.usage_over_time(result), flagged),
        charts.usage_by_type(metrics.usage_by_device_type(result)),
        charts.hourly_heatmap(metrics.hourly_profile(result)),
        charts.anomaly_scatter(result),
        charts.voltage_band(result, 220.0, 240.0),
    ]
    for figure in figures:
        assert isinstance(figure, go.Figure)
        assert figure.data


def test_timeline_adds_an_anomaly_overlay(frame: pd.DataFrame) -> None:
    result = detect(frame)
    flagged = result[result["detected_anomaly"] == 1]
    figure = charts.usage_timeline(metrics.usage_over_time(result), flagged)
    assert any(trace.name == "Anomaly" for trace in figure.data)


def test_charts_degrade_gracefully_when_there_is_nothing_to_plot() -> None:
    empty = pd.DataFrame(columns=["timestamp", "device_type", "usage_kwh"])
    figure = charts.usage_timeline(empty)
    assert figure.layout.annotations
    assert not figure.data
