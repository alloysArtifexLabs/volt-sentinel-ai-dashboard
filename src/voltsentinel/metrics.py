"""Fleet-level aggregations that feed the dashboard's KPIs and tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FleetSummary:
    """Headline numbers for the current selection."""

    total_usage_kwh: float
    average_load_kwh: float
    peak_load_kwh: float
    average_voltage: float
    reading_count: int
    device_count: int
    anomaly_count: int
    anomaly_rate: float
    uptime_pct: float
    busiest_device_type: str


def summarise(df: pd.DataFrame) -> FleetSummary:
    """Reduce a (possibly filtered) frame to its headline metrics."""
    if df.empty:
        return FleetSummary(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, "—")

    anomalies = int(df.get("detected_anomaly", pd.Series(dtype=int)).sum())
    by_type = df.groupby("device_type")["usage_kwh"].sum()

    return FleetSummary(
        total_usage_kwh=float(df["usage_kwh"].sum()),
        average_load_kwh=float(df["usage_kwh"].mean()),
        peak_load_kwh=float(df["usage_kwh"].max()),
        average_voltage=float(df["voltage"].mean()),
        reading_count=len(df),
        device_count=int(df["device_id"].nunique()),
        anomaly_count=anomalies,
        anomaly_rate=anomalies / len(df) * 100,
        uptime_pct=float((df["status"] == "ON").mean() * 100),
        busiest_device_type=str(by_type.idxmax()),
    )


def usage_over_time(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Average load per device type, resampled onto a regular grid.

    The mean — rather than the sum — keeps the y-scale identical at every
    granularity, so individual readings (the anomaly overlay) can be drawn on
    the same axis as the rolled-up trend.
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "device_type", "usage_kwh"])
    return (
        df.set_index("timestamp")
        .groupby("device_type")["usage_kwh"]
        .resample(freq)
        .mean()
        .dropna()
        .reset_index()
    )


def usage_by_device_type(df: pd.DataFrame) -> pd.DataFrame:
    """Total and mean usage plus anomaly counts for each device type."""
    if df.empty:
        return pd.DataFrame(
            columns=["device_type", "total_kwh", "mean_kwh", "anomalies", "readings"]
        )
    grouped = df.groupby("device_type").agg(
        total_kwh=("usage_kwh", "sum"),
        mean_kwh=("usage_kwh", "mean"),
        anomalies=("detected_anomaly", "sum"),
        readings=("usage_kwh", "size"),
    )
    return grouped.reset_index().sort_values("total_kwh", ascending=False)


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average load for each (device type, hour-of-day) pair."""
    if df.empty:
        return pd.DataFrame(columns=["device_type", "hour", "usage_kwh"])
    return (
        df.groupby(["device_type", "hour"])["usage_kwh"]
        .mean()
        .reset_index()
        .sort_values(["device_type", "hour"])
    )


def device_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Per-device rollup, ranked by the anomalies each device produced."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "device_id",
                "device_type",
                "total_kwh",
                "mean_voltage",
                "anomalies",
                "anomaly_rate_pct",
            ]
        )
    grouped = df.groupby(["device_id", "device_type"]).agg(
        total_kwh=("usage_kwh", "sum"),
        mean_voltage=("voltage", "mean"),
        anomalies=("detected_anomaly", "sum"),
        readings=("usage_kwh", "size"),
    )
    grouped["anomaly_rate_pct"] = grouped["anomalies"] / grouped["readings"] * 100
    return (
        grouped.drop(columns="readings")
        .reset_index()
        .sort_values(["anomalies", "total_kwh"], ascending=False)
    )


def anomaly_table(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """The flagged readings, most suspicious first."""
    if df.empty or "detected_anomaly" not in df.columns:
        return pd.DataFrame()
    columns = [
        "timestamp",
        "device_id",
        "device_type",
        "usage_kwh",
        "voltage",
        "status",
        "anomaly_score",
        "anomaly_reasons",
    ]
    available = [column for column in columns if column in df.columns]
    flagged = df[df["detected_anomaly"] == 1][available].sort_values(
        "anomaly_score", ascending=False
    )
    return flagged.head(limit) if limit else flagged
