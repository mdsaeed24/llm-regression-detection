"""Result models for an evaluation run, and the aggregate metrics computed from them."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from regress.dataset import Difficulty
from regress.models import CATEGORIES, Category
from regress.providers.base import TokenUsage

RunKind = Literal["baseline", "run"]
Status = Literal["PASS", "WARN", "FAIL"]


class CaseResult(BaseModel):
    """Every scored dimension for one golden case in one run.

    A case that errored (provider failure, unparseable output, judge failure) is
    recorded with ``passed=False`` and an ``error`` - it is never dropped, so a
    prompt that breaks the output format shows up as failures, not as a smaller
    and better-looking denominator.
    """

    case_id: str
    expected_category: Category
    expected_difficulty: Difficulty
    predicted_category: Category | None = None
    summary: str | None = None
    category_match: bool = False
    relevance_score: float | None = Field(default=None, ge=1.0, le=5.0)
    relevance_reason: str | None = None
    passed: bool = False
    error: str | None = None
    attempts: int = 0
    latency_ms: float = 0.0
    usage: TokenUsage = TokenUsage()
    cost_usd: float = 0.0
    judge_usage: TokenUsage = TokenUsage()
    judge_cost_usd: float = 0.0
    raw_outputs: list[str] = Field(default_factory=list)


class RunAggregates(BaseModel):
    """Run-level metrics. Rates are fractions in [0, 1]."""

    case_count: int
    passed_count: int
    pass_rate: float
    accuracy: float
    per_category_accuracy: dict[str, float]
    avg_relevance: float | None
    error_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    judge_cost_usd: float


class RunRecord(BaseModel):
    """Metadata for one run: everything needed to reproduce or explain it."""

    run_id: str
    created_at: datetime
    kind: RunKind
    prompt_version: str
    prompt_hash: str
    model: str
    dataset_version: int
    dataset_hash: str
    git_sha: str | None
    aggregates: RunAggregates
    baseline_run_id: str | None = None
    status: Status | None = None


class EvalRun(BaseModel):
    """A run's metadata plus its per-case results."""

    record: RunRecord
    results: list[CaseResult]


def compute_aggregates(results: list[CaseResult]) -> RunAggregates:
    """Aggregate per-case results. Errored cases count as incorrect, never as skipped."""
    n = len(results)
    if n == 0:
        raise ValueError("cannot aggregate an empty result set")

    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_category[r.expected_category].append(r.category_match)

    relevance = [r.relevance_score for r in results if r.relevance_score is not None]
    latencies = sorted(r.latency_ms for r in results)

    return RunAggregates(
        case_count=n,
        passed_count=sum(r.passed for r in results),
        pass_rate=sum(r.passed for r in results) / n,
        accuracy=sum(r.category_match for r in results) / n,
        per_category_accuracy={
            c: sum(by_category[c]) / len(by_category[c]) for c in CATEGORIES if by_category[c]
        },
        avg_relevance=statistics.fmean(relevance) if relevance else None,
        error_count=sum(r.error is not None for r in results),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        input_tokens=sum(r.usage.input_total for r in results),
        output_tokens=sum(r.usage.output for r in results),
        cost_usd=sum(r.cost_usd for r in results),
        judge_cost_usd=sum(r.judge_cost_usd for r in results),
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile; fine for tens of cases and has no numpy dependency."""
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]
