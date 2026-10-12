"""SQLite run history: one row per run, one row per case result.

Each table keeps the fields people filter and chart on as real columns (so the
dashboard and ad-hoc SQL stay simple) and the complete Pydantic model as JSON
(so nothing is lost and models round-trip exactly).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from regress.eval.results import CaseResult, EvalRun, RunKind, RunRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,          -- ISO 8601 UTC
    kind             TEXT NOT NULL CHECK (kind IN ('baseline', 'run')),
    prompt_version   TEXT NOT NULL,
    prompt_hash      TEXT NOT NULL,
    model            TEXT NOT NULL,
    dataset_version  INTEGER NOT NULL,
    dataset_hash     TEXT NOT NULL,
    git_sha          TEXT,
    case_count       INTEGER NOT NULL,
    accuracy         REAL NOT NULL,
    pass_rate        REAL NOT NULL,
    avg_relevance    REAL,
    total_cost_usd   REAL NOT NULL,          -- feature + judge
    baseline_run_id  TEXT REFERENCES runs (run_id),
    status           TEXT CHECK (status IN ('PASS', 'WARN', 'FAIL')),
    aggregates_json  TEXT NOT NULL,
    record_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_kind_created ON runs (kind, created_at);

CREATE TABLE IF NOT EXISTS results (
    run_id              TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    case_id             TEXT NOT NULL,
    expected_category   TEXT NOT NULL,
    expected_difficulty TEXT NOT NULL,
    predicted_category  TEXT,
    category_match      INTEGER NOT NULL,
    relevance_score     REAL,
    passed              INTEGER NOT NULL,
    error               TEXT,
    latency_ms          REAL NOT NULL,
    total_cost_usd      REAL NOT NULL,       -- feature + judge
    result_json         TEXT NOT NULL,       -- full CaseResult incl. raw model outputs
    PRIMARY KEY (run_id, case_id)
);
"""


class RunStore:
    """Persist and load ``EvalRun`` objects."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    def save(self, run: EvalRun) -> None:
        """Insert a run and all its results in one transaction."""
        r, agg = run.record, run.record.aggregates
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    r.run_id, r.created_at.isoformat(), r.kind, r.prompt_version, r.prompt_hash,
                    r.model, r.dataset_version, r.dataset_hash, r.git_sha, agg.case_count,
                    agg.accuracy, agg.pass_rate, agg.avg_relevance, agg.cost_usd + agg.judge_cost_usd,
                    r.baseline_run_id, r.status, agg.model_dump_json(), r.model_dump_json(),
                ),
            )  # fmt: skip
            conn.executemany(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        r.run_id, c.case_id, c.expected_category, c.expected_difficulty,
                        c.predicted_category, c.category_match, c.relevance_score, c.passed,
                        c.error, c.latency_ms, c.cost_usd + c.judge_cost_usd, c.model_dump_json(),
                    )  # fmt: skip
                    for c in run.results
                ],
            )

    def load(self, run_id: str) -> EvalRun:
        """Load one run with its results; raises ``KeyError`` if it does not exist."""
        with self._connect() as conn:
            row = conn.execute("SELECT record_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"No run with id '{run_id}' in {self._path}")
            results = conn.execute(
                "SELECT result_json FROM results WHERE run_id = ? ORDER BY case_id", (run_id,)
            ).fetchall()
        return EvalRun(
            record=RunRecord.model_validate_json(row["record_json"]),
            results=[CaseResult.model_validate_json(r["result_json"]) for r in results],
        )

    def latest(self, kind: RunKind | None = None) -> EvalRun | None:
        """Most recent run, optionally of one kind ('baseline' for --baseline latest)."""
        query = "SELECT run_id FROM runs"
        params: tuple[str, ...] = ()
        if kind:
            query += " WHERE kind = ?"
            params = (kind,)
        with self._connect() as conn:
            row = conn.execute(query + " ORDER BY created_at DESC LIMIT 1", params).fetchone()
        return self.load(row["run_id"]) if row else None

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        """Newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_json FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RunRecord.model_validate_json(r["record_json"]) for r in rows]
