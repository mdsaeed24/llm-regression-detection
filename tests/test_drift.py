"""Slow drift: moving average over shipped runs on the same dataset."""

import pytest

from regress.config import DriftSettings
from regress.eval.drift import check_drift
from tests.conftest import make_record

SETTINGS = DriftSettings(window=3, accuracy_floor=0.80)


def test_needs_a_full_window_before_it_can_fire():
    history = [make_record("r1", 0.10, minutes=1)]

    result = check_drift(history, make_record("now", 0.10, minutes=2), SETTINGS)

    assert not result.triggered
    assert result.moving_average is None
    assert "not enough history (2/3" in result.message


def test_fires_when_the_average_falls_below_the_floor():
    # Each step is small (no single-run alarm), but the trend crosses the floor.
    history = [make_record("r1", 0.82, 1), make_record("r2", 0.80, 2)]

    result = check_drift(history, make_record("now", 0.77, 3), SETTINGS)

    assert result.triggered
    assert result.moving_average == pytest.approx((0.82 + 0.80 + 0.77) / 3)


def test_does_not_fire_above_the_floor():
    history = [make_record("r1", 0.85, 1), make_record("r2", 0.84, 2)]

    assert not check_drift(history, make_record("now", 0.83, 3), SETTINGS).triggered


def test_only_the_most_recent_window_counts():
    # Old bad runs have rolled out of the 3-run window.
    history = [make_record(f"old{i}", 0.10, i) for i in range(5)]
    history += [make_record("r1", 0.90, 10), make_record("r2", 0.90, 11)]

    result = check_drift(history, make_record("now", 0.90, 12), SETTINGS)

    assert not result.triggered
    assert result.moving_average == pytest.approx(0.90)


def test_failed_runs_are_excluded_because_they_never_shipped():
    history = [make_record("r1", 0.90, 1), make_record("r2", 0.90, 2),
               make_record("blocked", 0.05, 3, status="FAIL")]  # fmt: skip

    result = check_drift(history, make_record("now", 0.90, 4), SETTINGS)

    assert result.moving_average == pytest.approx(0.90)


def test_other_dataset_versions_are_excluded():
    history = [make_record("r1", 0.90, 1), make_record("r2", 0.90, 2),
               make_record("other-dataset", 0.05, 3, dataset_version=2)]  # fmt: skip

    result = check_drift(history, make_record("now", 0.90, 4), SETTINGS)

    assert result.moving_average == pytest.approx(0.90)
    assert result.runs_considered == 3
