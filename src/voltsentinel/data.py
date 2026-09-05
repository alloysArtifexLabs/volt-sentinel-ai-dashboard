"""Loading, validation and normalisation of energy telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from voltsentinel.config import REQUIRED_COLUMNS, Settings, settings


class DataValidationError(ValueError):
    """Raised when a dataset is missing columns or cannot be parsed."""


def validate(df: pd.DataFrame) -> None:
    """Check that ``df`` carries every column the dashboard depends on."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError("Dataset is missing required column(s): " + ", ".join(missing))
    if df.empty:
        raise DataValidationError("Dataset contains no rows.")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, derive time features and drop unusable rows.

    The raw exports carry timestamps as strings and occasionally include
    readings with no usage value; both are handled here so that every
    downstream module can assume clean numeric columns.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in ("usage_kwh", "voltage"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["timestamp", "usage_kwh", "voltage"])
    if df.empty:
        raise DataValidationError(
            "No usable rows left after parsing timestamps and numeric columns."
        )

    df["status"] = df["status"].astype(str).str.strip().str.upper()
    df["device_type"] = df["device_type"].astype(str).str.strip()
    df["device_id"] = df["device_id"].astype(str).str.strip()

    # Derive time features rather than trusting the exported `hour` column,
    # which drifts out of sync whenever rows are appended by hand.
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["day_name"] = df["timestamp"].dt.day_name()

    # Keep the vendor's pre-computed flags as a baseline to compare against.
    for legacy, renamed in (
        ("anomaly_flag", "baseline_rule_flag"),
        ("ai_anomaly_flag", "baseline_ai_flag"),
    ):
        if legacy in df.columns:
            df[renamed] = pd.to_numeric(df[legacy], errors="coerce").fillna(0).astype(int)

    return df.sort_values("timestamp").reset_index(drop=True)


def load_dataset(source: str | Path | IO[bytes] | None = None) -> pd.DataFrame:
    """Read, validate and normalise a telemetry CSV.

    ``source`` may be a path, an open file object (e.g. a Streamlit upload)
    or ``None`` to fall back to the bundled dataset.
    """
    if source is None:
        source = settings.data_path

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"Dataset not found at {path}")
        raw = pd.read_csv(path)
    else:
        raw = pd.read_csv(source)

    validate(raw)
    return normalise(raw)


def describe_source(config: Settings | None = None) -> str:
    """Human-readable label for the dataset currently configured."""
    config = config or settings
    return str(config.data_path)
