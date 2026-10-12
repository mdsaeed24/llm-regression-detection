"""The committed baseline: baselines/baseline.json.

CI runners start empty, so the SQLite history does not survive between jobs.
The baseline therefore lives in git: the full baseline run (so regressed cases
can be shown side by side) plus the records of earlier accepted baselines (so
slow drift has history to average over).

Updating it is a deliberate, reviewed change: re-run the baseline and commit the
file in the PR. CI always compares against the *base branch's* copy, so a PR
cannot hide its own regression by also updating the baseline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from regress.eval.results import EvalRun, RunRecord
from regress.storage.db import RunStore

DEFAULT_HISTORY_KEEP = 30


class BaselineFile(BaseModel):
    schema_version: Literal[1] = 1
    exported_at: datetime
    baseline: EvalRun
    history: list[RunRecord]  # earlier accepted baselines, newest first


def load_baseline_file(path: Path | str) -> BaselineFile:
    return BaselineFile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def export_baseline(run: EvalRun, path: Path | str, keep: int = DEFAULT_HISTORY_KEEP) -> BaselineFile:
    """Write ``run`` as the new baseline, chaining the previous file's history."""
    path = Path(path)
    history: list[RunRecord] = []
    if path.exists():
        previous = load_baseline_file(path)
        history = [previous.baseline.record, *previous.history]
    history = [r for r in history if r.run_id != run.record.run_id][:keep]
    baseline_file = BaselineFile(exported_at=datetime.now(UTC), baseline=run, history=history)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline_file.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return baseline_file


def import_baseline(baseline_file: BaselineFile, store: RunStore) -> EvalRun:
    """Load the file's runs into ``store`` (idempotent) and return the baseline run."""
    if not store.exists(baseline_file.baseline.record.run_id):
        store.save(baseline_file.baseline)
    for record in baseline_file.history:
        if not store.exists(record.run_id):
            store.save_record(record)
    return baseline_file.baseline
