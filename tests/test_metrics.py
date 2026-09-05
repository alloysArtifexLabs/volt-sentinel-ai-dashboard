from __future__ import annotations

import pandas as pd

from voltsentinel import metrics
from voltsentinel.detection import detect


def scored(frame: pd.DataFrame) -> pd.DataFrame:
    return detect(frame)


def test_summary_matches_the_underlying_frame(frame: pd.DataFrame) -> None:
    result = scored(frame)
    summary = metrics.summarise(result)
    assert summary.reading_count == len(result)
    assert summary.device_count == result["device_id"].nunique()
    assert summary.total_usage_kwh == float(result["usage_kwh"].sum())
    assert 0 <= summary.anomaly_rate <= 100
    assert 0 <= summary.uptime_pct <= 100


def test_summary_of_an_empty_frame_is_all_zeroes() -> None:
    summary = metrics.summarise(pd.DataFrame())
    assert summary.reading_count == 0
    assert summary.busiest_device_type == "—"


def test_hourly_resampling_is_a_no_op_for_hourly_data(frame: pd.DataFrame) -> None:
    # One reading per device per hour, so the hourly mean is the reading itself.
    resampled = metrics.usage_over_time(scored(frame), "h")
    assert len(resampled) == len(frame)
    assert round(resampled["usage_kwh"].sum(), 6) == round(frame["usage_kwh"].sum(), 6)


def test_coarser_granularity_keeps_the_same_scale(frame: pd.DataFrame) -> None:
    result = scored(frame)
    hourly = metrics.usage_over_time(result, "h")
    daily = metrics.usage_over_time(result, "D")
    # Averaging, not summing: a daily roll-up must not inflate the y-axis, or
    # the raw-reading anomaly overlay would sit on a different scale.
    assert len(daily) < len(hourly)
    assert daily["usage_kwh"].max() <= hourly["usage_kwh"].max()


def test_usage_by_device_type_is_ranked_by_total(frame: pd.DataFrame) -> None:
    summary = metrics.usage_by_device_type(scored(frame))
    assert summary["total_kwh"].is_monotonic_decreasing
    assert summary["readings"].sum() == len(frame)


def test_hourly_profile_covers_each_type_and_hour(frame: pd.DataFrame) -> None:
    profile = metrics.hourly_profile(scored(frame))
    expected = frame["device_type"].nunique() * frame["hour"].nunique()
    assert len(profile) == expected


def test_device_leaderboard_reports_a_percentage_rate(frame: pd.DataFrame) -> None:
    leaderboard = metrics.device_leaderboard(scored(frame))
    assert leaderboard["anomaly_rate_pct"].between(0, 100).all()
    assert leaderboard["anomalies"].is_monotonic_decreasing


def test_anomaly_table_is_sorted_by_score(frame: pd.DataFrame) -> None:
    table = metrics.anomaly_table(scored(frame))
    assert not table.empty
    assert table["anomaly_score"].is_monotonic_decreasing


def test_anomaly_table_honours_the_limit(frame: pd.DataFrame) -> None:
    assert len(metrics.anomaly_table(scored(frame), limit=2)) == 2


def test_aggregations_survive_an_empty_frame() -> None:
    empty = pd.DataFrame()
    assert metrics.usage_over_time(empty).empty
    assert metrics.usage_by_device_type(empty).empty
    assert metrics.hourly_profile(empty).empty
    assert metrics.device_leaderboard(empty).empty
    assert metrics.anomaly_table(empty).empty
