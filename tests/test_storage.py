"""SQLite run history and the committed baseline file."""

import pytest

from regress.storage import RunStore, export_baseline, import_baseline, load_baseline_file
from tests.conftest import make_result, make_run


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(tmp_path / "runs.sqlite")


def test_round_trip_preserves_every_field(store):
    run = make_run([make_result("a"), make_result("b", predicted="general")], "r1", status="WARN")

    store.save(run)

    assert store.load("r1") == run


def test_latest_baseline_ignores_newer_runs(store):
    store.save(make_run([make_result("a")], "base-old", kind="baseline", minutes=1))
    store.save(make_run([make_result("a")], "base-new", kind="baseline", minutes=2))
    store.save(make_run([make_result("a")], "run-newest", kind="run", minutes=3))

    assert store.latest("baseline").record.run_id == "base-new"
    assert store.latest().record.run_id == "run-newest"
    assert [r.run_id for r in store.list_runs()] == ["run-newest", "base-new", "base-old"]


def test_missing_run_raises_key_error(store):
    with pytest.raises(KeyError, match="nope"):
        store.load("nope")


def test_baseline_file_chains_history_and_imports_idempotently(tmp_path, store):
    path = tmp_path / "baselines" / "baseline.json"
    first = make_run([make_result("a")], "b1", kind="baseline", minutes=1)
    second = make_run([make_result("a")], "b2", kind="baseline", minutes=2)

    export_baseline(first, path)
    exported = export_baseline(second, path)
    export_baseline(second, path)  # re-exporting the same run must not duplicate it

    loaded = load_baseline_file(path)
    assert exported.baseline.record.run_id == "b2"
    assert [r.run_id for r in loaded.history] == ["b1"]

    for _ in range(2):  # importing twice is a no-op the second time
        baseline = import_baseline(loaded, store)
    assert baseline.record.run_id == "b2"
    assert store.exists("b1") and store.exists("b2")
    assert len(store.list_runs()) == 2
