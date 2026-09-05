from __future__ import annotations

import numpy as np
import pandas as pd

from voltsentinel.config import DETECTION_COLUMNS, Settings
from voltsentinel.detection import (
    build_features,
    detect,
    isolation_forest_flags,
    robust_zscore,
    rule_flags,
    voltage_flags,
    zscore_flags,
)


def test_detect_appends_every_expected_column(frame: pd.DataFrame, config: Settings) -> None:
    result = detect(frame, config)
    for column in DETECTION_COLUMNS:
        assert column in result.columns


def test_detect_does_not_mutate_the_input(frame: pd.DataFrame, config: Settings) -> None:
    before = frame.copy()
    detect(frame, config)
    pd.testing.assert_frame_equal(frame, before)


def test_phantom_load_is_flagged(frame: pd.DataFrame, config: Settings) -> None:
    flags = rule_flags(frame, config)
    offenders = frame[flags == 1]
    assert len(offenders) == 1
    assert offenders.iloc[0]["status"] == "OFF"
    assert offenders.iloc[0]["usage_kwh"] > config.phantom_load_kwh


def test_voltage_outside_the_band_is_flagged(frame: pd.DataFrame, config: Settings) -> None:
    flagged = frame[voltage_flags(frame, config) == 1]
    assert (flagged["voltage"] == 190.0).any()


def test_robust_zscore_is_not_dragged_by_its_own_outlier() -> None:
    baseline = list(np.linspace(48.0, 52.0, 50))
    robust = robust_zscore(pd.Series([*baseline, 1000.0]))
    classic = pd.Series([*baseline, 1000.0])
    classic = (classic - classic.mean()) / classic.std(ddof=0)
    # The median/MAD score puts the outlier far out; the mean/std one cannot,
    # because the outlier inflated the very spread it is measured against.
    assert abs(robust.iloc[-1]) > 100
    assert abs(classic.iloc[-1]) < 8
    assert robust.iloc[:-1].abs().max() < 3.5


def test_robust_zscore_falls_back_when_the_mad_collapses() -> None:
    # A near-constant series has a zero MAD; the standard deviation stands in.
    scores = robust_zscore(pd.Series([10.0] * 50 + [1000.0]))
    assert scores.iloc[-1] > 0
    assert scores.iloc[:-1].abs().max() < 1.0


def test_robust_zscore_survives_a_constant_series() -> None:
    scores = robust_zscore(pd.Series([5.0] * 20))
    assert (scores == 0).all()


def test_zscore_is_computed_per_device_type(frame: pd.DataFrame, config: Settings) -> None:
    shifted = frame.copy()
    # Lift one device type far above the rest; per-type scoring should ignore
    # the level shift, whereas a global z-score would flag the whole group.
    shifted.loc[shifted["device_type"] == "Pump", "usage_kwh"] += 500
    _, flags = zscore_flags(shifted, config)
    assert flags[shifted["device_type"] == "Pump"].sum() == 0


def test_cyclic_hour_features_wrap_around(frame: pd.DataFrame) -> None:
    features = build_features(frame)
    assert set(features.columns) == {"usage_kwh", "voltage", "hour_sin", "hour_cos"}
    magnitudes = np.hypot(features["hour_sin"], features["hour_cos"])
    np.testing.assert_allclose(magnitudes, 1.0, atol=1e-9)


def test_features_exclude_on_off_state(frame: pd.DataFrame) -> None:
    # A rare binary scales to several sigma — further out than any real load or
    # voltage excursion — so including it would let the forest isolate every
    # OFF reading in one split. Status is the rule detector's job.
    assert "is_on" not in build_features(frame).columns


def test_isolation_forest_is_not_just_a_status_detector(bundled: pd.DataFrame) -> None:
    """Regression: the forest once flagged 26 OFF rows and zero ON rows, so
    deselecting OFF in the dashboard reported no anomalies across 771 readings."""
    result = detect(bundled)
    flagged = result[result["iforest_flag"] == 1]
    assert not flagged.empty
    # Most readings are ON, so most flags should be too.
    assert (flagged["status"] == "ON").sum() > (flagged["status"] != "ON").sum()

    on_only = result[result["status"] == "ON"]
    assert on_only["detected_anomaly"].sum() > 0, (
        "filtering to ON-only devices must not zero out the anomaly count"
    )


def test_isolation_forest_scores_are_normalised(frame: pd.DataFrame, config: Settings) -> None:
    scores, flags = isolation_forest_flags(frame, config)
    assert scores.between(0, 1).all()
    assert set(flags.unique()) <= {0, 1}
    assert flags.sum() > 0


def test_isolation_forest_is_deterministic(frame: pd.DataFrame, config: Settings) -> None:
    first, _ = isolation_forest_flags(frame, config)
    second, _ = isolation_forest_flags(frame, config)
    pd.testing.assert_series_equal(first, second)


def test_tiny_frames_do_not_fit_a_model(frame: pd.DataFrame, config: Settings) -> None:
    scores, flags = isolation_forest_flags(frame.head(5), config)
    assert (scores == 0).all()
    assert (flags == 0).all()


def test_anomaly_score_stays_within_bounds(frame: pd.DataFrame, config: Settings) -> None:
    result = detect(frame, config)
    assert result["anomaly_score"].between(0, 1).all()


def test_reasons_are_populated_only_for_flagged_rows(frame: pd.DataFrame, config: Settings) -> None:
    result = detect(frame, config)
    assert (result.loc[result["detected_anomaly"] == 0, "anomaly_reasons"] == "—").all()
    assert (result.loc[result["detected_anomaly"] == 1, "anomaly_reasons"] != "—").all()


def test_a_stricter_contamination_flags_no_more_readings(
    frame: pd.DataFrame, config: Settings
) -> None:
    loose = Settings(data_path=config.data_path, contamination=0.10)
    tight = Settings(data_path=config.data_path, contamination=0.01)
    assert detect(frame, tight)["iforest_flag"].sum() <= detect(frame, loose)["iforest_flag"].sum()


def test_bundled_dataset_produces_anomalies(bundled: pd.DataFrame) -> None:
    result = detect(bundled)
    assert 0 < result["detected_anomaly"].sum() < len(result)
