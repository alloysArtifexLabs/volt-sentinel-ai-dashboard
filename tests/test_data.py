from __future__ import annotations

import pandas as pd
import pytest

from voltsentinel.config import REQUIRED_COLUMNS
from voltsentinel.data import DataValidationError, load_dataset, normalise, validate


def test_bundled_dataset_loads(bundled: pd.DataFrame) -> None:
    assert not bundled.empty
    for column in REQUIRED_COLUMNS:
        assert column in bundled.columns


def test_timestamps_are_parsed_and_sorted(bundled: pd.DataFrame) -> None:
    assert pd.api.types.is_datetime64_any_dtype(bundled["timestamp"])
    assert bundled["timestamp"].is_monotonic_increasing


def test_hour_is_derived_from_the_timestamp(frame: pd.DataFrame) -> None:
    assert (frame["hour"] == frame["timestamp"].dt.hour).all()


def test_legacy_flags_are_preserved_under_baseline_names(bundled: pd.DataFrame) -> None:
    assert "baseline_rule_flag" in bundled.columns
    assert "baseline_ai_flag" in bundled.columns


def test_status_is_normalised(raw_frame: pd.DataFrame) -> None:
    raw_frame = raw_frame.copy()
    raw_frame.loc[2, "status"] = " on "
    assert normalise(raw_frame).loc[lambda d: d.index == 2, "status"].iloc[0] == "ON"


def test_validate_rejects_missing_columns() -> None:
    with pytest.raises(DataValidationError, match="missing required column"):
        validate(pd.DataFrame({"timestamp": ["2025-01-01"]}))


def test_validate_rejects_empty_frames() -> None:
    with pytest.raises(DataValidationError, match="no rows"):
        validate(pd.DataFrame({column: [] for column in REQUIRED_COLUMNS}))


def test_unparseable_rows_are_dropped(raw_frame: pd.DataFrame) -> None:
    dirty = raw_frame.copy()
    dirty["usage_kwh"] = dirty["usage_kwh"].astype(object)
    dirty.loc[3, "usage_kwh"] = "not-a-number"
    assert len(normalise(dirty)) == len(raw_frame) - 1


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(DataValidationError, match="not found"):
        load_dataset(tmp_path / "nope.csv")
