"""Anomaly detection for energy telemetry.

Three complementary detectors run over every reading:

``rule``
    Domain rules that encode what an engineer would spot by eye — a device
    reporting ``OFF`` while still drawing power, or a supply voltage outside
    its acceptable band.
``zscore``
    A robust (median / MAD) z-score computed *per device type*, so a server
    rack is compared against other server racks rather than against lighting.
``iforest``
    An Isolation Forest over usage, voltage and cyclic time-of-day features,
    which catches combinations no single-column threshold would. It sees only
    continuous signal: on/off state is deliberately excluded (see
    ``build_features``).

Each detector contributes to ``anomaly_score`` (0-1), and ``detected_anomaly``
is set when any of them fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from voltsentinel.config import Settings, settings

FEATURE_COLUMNS = ["usage_kwh", "voltage", "hour_sin", "hour_cos"]

#: Human-readable explanation for each detector, surfaced in the UI.
DETECTOR_LABELS: dict[str, str] = {
    "rule_flag": "Phantom load (drawing power while OFF)",
    "voltage_flag": "Voltage outside acceptable band",
    "zscore_flag": "Statistical outlier for its device type",
    "iforest_flag": "Isolation Forest outlier",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw readings into the numeric matrix the model consumes.

    Only continuous signal goes in. On/off state is deliberately excluded: it
    is a rare binary (roughly 8% of readings are OFF), so once scaled it lands
    several sigma from the centroid — further out than any genuine load or
    voltage excursion reaches. The forest would isolate every OFF reading in a
    single split and spend its whole contamination budget there, rediscovering
    ``status == "OFF"`` instead of finding unusual energy behaviour, and
    reporting zero anomalies whenever OFF readings are filtered out. Phantom
    loads are caught explicitly by :func:`rule_flags` instead.
    """
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    features = pd.DataFrame(index=df.index)
    features["usage_kwh"] = df["usage_kwh"].astype(float)
    features["voltage"] = df["voltage"].astype(float)
    # Encode the clock as a circle so 23:00 and 00:00 sit next to each other.
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return features[FEATURE_COLUMNS]


def robust_zscore(series: pd.Series) -> pd.Series:
    """Median/MAD z-score — unlike the mean/std version it is not dragged
    around by the very outliers it is meant to find."""
    values = series.astype(float)
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0 or np.isnan(mad):
        spread = values.std(ddof=0)
        if not spread or np.isnan(spread):
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values - median) / spread
    # 0.6745 rescales the MAD to be comparable with a standard deviation.
    return 0.6745 * (values - median) / mad


def rule_flags(df: pd.DataFrame, config: Settings | None = None) -> pd.Series:
    """Flag devices reporting ``OFF`` while still drawing significant power."""
    config = config or settings
    return ((df["status"] != "ON") & (df["usage_kwh"] > config.phantom_load_kwh)).astype(int)


def voltage_flags(df: pd.DataFrame, config: Settings | None = None) -> pd.Series:
    """Flag readings whose supply voltage sits outside the acceptable band."""
    config = config or settings
    return ((df["voltage"] < config.voltage_min) | (df["voltage"] > config.voltage_max)).astype(int)


def zscore_flags(df: pd.DataFrame, config: Settings | None = None) -> tuple[pd.Series, pd.Series]:
    """Return ``(zscore, flag)`` computed per device type."""
    config = config or settings
    scores = df.groupby("device_type")["usage_kwh"].transform(robust_zscore).fillna(0.0)
    return scores, (scores.abs() > config.zscore_threshold).astype(int)


def isolation_forest_flags(
    df: pd.DataFrame, config: Settings | None = None
) -> tuple[pd.Series, pd.Series]:
    """Fit an Isolation Forest and return ``(score, flag)``.

    The score is normalised to 0-1 where 1 is the most anomalous reading, so
    it can be blended with the other detectors and shown on a colour scale.
    """
    config = config or settings
    features = build_features(df)

    # With a handful of readings the ensemble has nothing to learn from.
    if len(features) < 10:
        zeros = pd.Series(np.zeros(len(df)), index=df.index)
        return zeros, zeros.astype(int)

    scaled = StandardScaler().fit_transform(features)
    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    predictions = model.fit_predict(scaled)
    # decision_function: higher is more normal, so negate for an outlier score.
    raw_scores = -model.decision_function(scaled)

    span = raw_scores.max() - raw_scores.min()
    normalised = (raw_scores - raw_scores.min()) / span if span > 0 else np.zeros_like(raw_scores)

    return (
        pd.Series(normalised, index=df.index),
        pd.Series((predictions == -1).astype(int), index=df.index),
    )


def detect(df: pd.DataFrame, config: Settings | None = None) -> pd.DataFrame:
    """Run every detector and append their verdicts to ``df``.

    Returns a new frame; the input is never mutated.
    """
    config = config or settings
    result = df.copy()

    result["rule_flag"] = rule_flags(result, config)
    result["voltage_flag"] = voltage_flags(result, config)
    result["zscore"], result["zscore_flag"] = zscore_flags(result, config)
    result["iforest_score"], result["iforest_flag"] = isolation_forest_flags(result, config)

    flag_columns = list(DETECTOR_LABELS)
    result["detected_anomaly"] = (result[flag_columns].sum(axis=1) > 0).astype(int)

    # Blend the model score with how many detectors agree, so a reading that
    # trips three checks outranks one that only just crossed a threshold.
    agreement = result[flag_columns].sum(axis=1) / len(flag_columns)
    result["anomaly_score"] = (0.6 * result["iforest_score"] + 0.4 * agreement).clip(0, 1)

    result["anomaly_reasons"] = result.apply(
        lambda row: (
            ", ".join(label for column, label in DETECTOR_LABELS.items() if row[column] == 1) or "—"
        ),
        axis=1,
    )
    return result
