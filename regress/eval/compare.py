"""Compare a run against a baseline: metric deltas plus per-case regressions/improvements."""

from __future__ import annotations

from pydantic import BaseModel

from regress.dataset import Difficulty
from regress.eval.results import CaseResult, EvalRun, compute_aggregates
from regress.models import CATEGORIES, Category


class CaseSnapshot(BaseModel):
    """The parts of a case result shown side by side in a diff."""

    predicted_category: Category | None
    summary: str | None
    relevance_score: float | None
    passed: bool
    error: str | None


class CaseChange(BaseModel):
    """A case whose pass/fail outcome flipped between baseline and run."""

    case_id: str
    expected_category: Category
    expected_difficulty: Difficulty
    before: CaseSnapshot
    after: CaseSnapshot
    reason: str


class CategoryDelta(BaseModel):
    cases: int
    before: float
    after: float
    delta_pp: float


class Comparison(BaseModel):
    """Run vs baseline, computed on the cases both runs share.

    Restricting to shared case IDs keeps the comparison apples-to-apples when the
    dataset changed or one side ran with ``--limit``. Rates are fractions; deltas
    in percentage points (pp) are ``(after - before) * 100``.
    """

    baseline_run_id: str
    run_id: str
    baseline_dataset_version: int
    run_dataset_version: int
    compared_cases: int
    only_in_baseline: list[str]
    only_in_run: list[str]
    accuracy_before: float
    accuracy_after: float
    accuracy_delta_pp: float
    pass_rate_before: float
    pass_rate_after: float
    pass_rate_delta_pp: float
    avg_relevance_before: float | None
    avg_relevance_after: float | None
    avg_relevance_delta: float | None
    per_category: dict[str, CategoryDelta]
    regressions: list[CaseChange]
    improvements: list[CaseChange]

    @property
    def dataset_mismatch(self) -> bool:
        return self.baseline_dataset_version != self.run_dataset_version


def compare_runs(baseline: EvalRun, run: EvalRun) -> Comparison:
    """Diff ``run`` against ``baseline``. Raises if they share no cases."""
    base_by_id = {r.case_id: r for r in baseline.results}
    run_by_id = {r.case_id: r for r in run.results}
    shared = sorted(base_by_id.keys() & run_by_id.keys())
    if not shared:
        raise ValueError(
            f"Run {run.record.run_id} and baseline {baseline.record.run_id} share no cases"
        )

    before = compute_aggregates([base_by_id[i] for i in shared])
    after = compute_aggregates([run_by_id[i] for i in shared])

    per_category: dict[str, CategoryDelta] = {}
    for category in CATEGORIES:
        if category in before.per_category_accuracy:
            b = before.per_category_accuracy[category]
            a = after.per_category_accuracy[category]
            per_category[category] = CategoryDelta(
                cases=sum(base_by_id[i].expected_category == category for i in shared),
                before=b,
                after=a,
                delta_pp=(a - b) * 100,
            )

    regressions, improvements = [], []
    for case_id in shared:
        b, a = base_by_id[case_id], run_by_id[case_id]
        if b.passed and not a.passed:
            regressions.append(_change(b, a))
        elif a.passed and not b.passed:
            improvements.append(_change(b, a))

    relevance_delta = (
        after.avg_relevance - before.avg_relevance
        if after.avg_relevance is not None and before.avg_relevance is not None
        else None
    )
    return Comparison(
        baseline_run_id=baseline.record.run_id,
        run_id=run.record.run_id,
        baseline_dataset_version=baseline.record.dataset_version,
        run_dataset_version=run.record.dataset_version,
        compared_cases=len(shared),
        only_in_baseline=sorted(base_by_id.keys() - run_by_id.keys()),
        only_in_run=sorted(run_by_id.keys() - base_by_id.keys()),
        accuracy_before=before.accuracy,
        accuracy_after=after.accuracy,
        accuracy_delta_pp=(after.accuracy - before.accuracy) * 100,
        pass_rate_before=before.pass_rate,
        pass_rate_after=after.pass_rate,
        pass_rate_delta_pp=(after.pass_rate - before.pass_rate) * 100,
        avg_relevance_before=before.avg_relevance,
        avg_relevance_after=after.avg_relevance,
        avg_relevance_delta=relevance_delta,
        per_category=per_category,
        regressions=regressions,
        improvements=improvements,
    )


def _snapshot(r: CaseResult) -> CaseSnapshot:
    return CaseSnapshot(
        predicted_category=r.predicted_category,
        summary=r.summary,
        relevance_score=r.relevance_score,
        passed=r.passed,
        error=r.error,
    )


def _change(before: CaseResult, after: CaseResult) -> CaseChange:
    return CaseChange(
        case_id=after.case_id,
        expected_category=after.expected_category,
        expected_difficulty=after.expected_difficulty,
        before=_snapshot(before),
        after=_snapshot(after),
        reason=_explain(before, after),
    )


def _explain(before: CaseResult, after: CaseResult) -> str:
    """Human-readable cause of a pass/fail flip, e.g. 'category billing -> general'."""
    if after.error and not before.error:
        return after.error  # the other fields are just missing because of it
    parts = []
    if before.predicted_category != after.predicted_category:
        parts.append(f"category {before.predicted_category} -> {after.predicted_category}")
    if before.relevance_score != after.relevance_score:
        parts.append(f"relevance {_fmt(before.relevance_score)} -> {_fmt(after.relevance_score)}")
    if after.error and after.error != before.error:
        parts.append(after.error)
    elif before.error and not after.error:
        parts.append("error resolved")
    return "; ".join(parts) or "outcome changed"


def _fmt(score: float | None) -> str:
    return "n/a" if score is None else f"{score:g}"
