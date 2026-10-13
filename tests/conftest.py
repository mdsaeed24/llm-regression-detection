"""Shared builders and a scripted fake LLM provider. No test makes a network call."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from regress.config import Settings, load_settings
from regress.eval.results import CaseResult, EvalRun, RunRecord, compute_aggregates
from regress.providers.base import LLMRequest, LLMResponse, TokenUsage

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def make_result(
    case_id: str,
    expected: str = "billing",
    predicted: str | None = "billing",
    relevance: float | None = 4.0,
    error: str | None = None,
    difficulty: str = "easy",
) -> CaseResult:
    """A scored case; ``passed`` follows the real rule (match AND relevance >= 3)."""
    match = predicted == expected
    return CaseResult(
        case_id=case_id,
        expected_category=expected,
        expected_difficulty=difficulty,
        predicted_category=predicted,
        summary=None if predicted is None else f"Customer summary for {case_id}.",
        category_match=match,
        relevance_score=relevance,
        passed=match and relevance is not None and relevance >= 3 and error is None,
        error=error,
    )


def make_run(
    results: list[CaseResult],
    run_id: str = "run",
    kind: str = "run",
    dataset_version: int = 1,
    minutes: int = 0,
    status: str | None = None,
) -> EvalRun:
    record = RunRecord(
        run_id=run_id,
        created_at=T0 + timedelta(minutes=minutes),
        kind=kind,
        prompt_version="v-test",
        prompt_hash="abc123",
        model="fake-model",
        dataset_version=dataset_version,
        dataset_hash="hash",
        git_sha=None,
        aggregates=compute_aggregates(results),
        status=status,
    )
    return EvalRun(record=record, results=results)


def make_record(run_id: str, accuracy: float, minutes: int, dataset_version: int = 1,
                status: str | None = "PASS") -> RunRecord:  # fmt: skip
    """A run record with a given accuracy (for drift tests)."""
    n = 100
    correct = round(accuracy * n)
    results = [make_result(f"c{i}", predicted="billing" if i < correct else "general") for i in range(n)]
    return make_run(results, run_id, dataset_version=dataset_version, minutes=minutes, status=status).record


class FakeProvider:
    """Scripted ``LLMProvider``: ``respond(request) -> text`` or raise. Records every request."""

    name = "fake"

    def __init__(self, respond: Callable[[LLMRequest], str]) -> None:
        self._respond = respond
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        text = self._respond(request)
        return LLMResponse(
            text=text,
            model=request.model,
            usage=TokenUsage(input_cache_miss=100, input_cache_hit=50, output=20),
            latency_ms=1.0,
        )


def classification_json(category: str, summary: str = "Customer reports a problem and wants it fixed.") -> str:
    return json.dumps({"category": category, "summary": summary})


def judge_json(score: int, reason: str = "fake judge verdict") -> str:
    return json.dumps({"reason": reason, "score": score})


@pytest.fixture
def settings() -> Settings:
    """The real config.yaml, so tests exercise the shipped thresholds and pricing."""
    return load_settings("config.yaml")
