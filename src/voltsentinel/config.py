"""Central configuration for VoltSentinel.

Every tunable lives here so the dashboard, the tests and any batch job all
agree on thresholds, column names and file locations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

#: Columns every dataset must provide before the dashboard will render it.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "device_id",
    "device_type",
    "usage_kwh",
    "voltage",
    "status",
)

#: Columns produced by the detection pipeline, appended to the raw frame.
DETECTION_COLUMNS: tuple[str, ...] = (
    "iforest_flag",
    "iforest_score",
    "zscore",
    "zscore_flag",
    "rule_flag",
    "voltage_flag",
    "anomaly_score",
    "detected_anomaly",
)


def _default_data_path() -> Path:
    """Resolve the bundled dataset, tolerating the pre-0.2 layout."""
    candidates = (
        PROJECT_ROOT / "data" / "simulated_energy_data_with_ai.csv",
        PROJECT_ROOT / "simulated_energy_data_with_ai.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class Settings:
    """Runtime settings, overridable through environment variables."""

    data_path: Path = field(default_factory=_default_data_path)

    # --- Detection tuning -------------------------------------------------
    #: Share of readings Isolation Forest is allowed to call anomalous.
    contamination: float = 0.03
    #: Number of trees in the Isolation Forest ensemble.
    n_estimators: int = 200
    #: Seed so a given dataset always yields the same anomalies.
    random_state: int = 42
    #: Robust z-score (median/MAD) above which a reading is an outlier.
    zscore_threshold: float = 3.5
    #: A device reporting OFF while drawing more than this is a phantom load.
    phantom_load_kwh: float = 90.0
    #: Acceptable supply voltage band; readings outside it are flagged.
    voltage_min: float = 220.0
    voltage_max: float = 240.0

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``VOLTSENTINEL_*`` environment variables."""

        def _get(name: str, cast, default):
            raw = os.environ.get(f"VOLTSENTINEL_{name.upper()}")
            if raw is None or raw == "":
                return default
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return default

        defaults = cls()
        return cls(
            data_path=Path(_get("data_path", str, defaults.data_path)),
            contamination=_get("contamination", float, defaults.contamination),
            n_estimators=_get("n_estimators", int, defaults.n_estimators),
            random_state=_get("random_state", int, defaults.random_state),
            zscore_threshold=_get("zscore_threshold", float, defaults.zscore_threshold),
            phantom_load_kwh=_get("phantom_load_kwh", float, defaults.phantom_load_kwh),
            voltage_min=_get("voltage_min", float, defaults.voltage_min),
            voltage_max=_get("voltage_max", float, defaults.voltage_max),
        )


settings = Settings.from_env()
