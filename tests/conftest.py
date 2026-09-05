"""Shared fixtures: a small synthetic fleet plus the bundled dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voltsentinel.config import Settings, settings
from voltsentinel.data import load_dataset, normalise


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """A deterministic fleet: three device types over two days, hourly."""
    rng = np.random.default_rng(7)
    timestamps = pd.date_range("2025-03-20", periods=48, freq="h")
    rows = []
    for device_index, device_type in enumerate(["HVAC", "Pump", "Lighting"], start=1):
        for timestamp in timestamps:
            rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "device_id": f"DEV-{device_index}",
                    "device_type": device_type,
                    "usage_kwh": round(float(rng.normal(50, 8)), 2),
                    "voltage": round(float(rng.normal(230, 3)), 2),
                    "status": "ON",
                    "anomaly_flag": 0,
                    "hour": timestamp.hour,
                    "ai_anomaly_flag": 0,
                }
            )
    frame = pd.DataFrame(rows)
    # One unmistakable phantom load: a device drawing hard while reporting OFF.
    frame.loc[0, ["usage_kwh", "status"]] = [140.0, "OFF"]
    # One under-voltage reading.
    frame.loc[1, "voltage"] = 190.0
    return frame


@pytest.fixture
def frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return normalise(raw_frame)


@pytest.fixture
def config() -> Settings:
    return Settings(data_path=settings.data_path)


@pytest.fixture(scope="session")
def bundled() -> pd.DataFrame:
    return load_dataset()
