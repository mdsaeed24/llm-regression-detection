"""Plain-text run summaries for the terminal (and CI logs)."""

from __future__ import annotations

from regress.eval.compare import CaseChange, Comparison
from regress.eval.results import EvalRun, RunRecord
from regress.eval.thresholds import Verdict

MAX_LISTED_CHANGES = 15


def format_run(run: EvalRun) -> str:
    """Header plus the run's own metrics (used for baselines and history)."""
    r, a = run.record, run.record.aggregates
    lines = [
        _header(r),
        f"  accuracy       {a.accuracy:7.1%}   ({sum(c.category_match for c in run.results)}/{a.case_count})",
        f"  pass rate      {a.pass_rate:7.1%}   ({a.passed_count}/{a.case_count})",
        f"  avg relevance  {_rel(a.avg_relevance):>7}   (1-5)",
    ]
    lines += [
        f"    {category:<10} {accuracy:7.1%}" for category, accuracy in a.per_category_accuracy.items()
    ]
    lines.append(_footer(run))
    return "\n".join(lines)


def format_comparison(run: EvalRun, comparison: Comparison, verdict: Verdict) -> str:
    """Scorecard vs baseline, flipped cases, and the verdict."""
    c = comparison
    lines = [
        _header(run.record),
        f"  baseline: {c.baseline_run_id}   compared cases: {c.compared_cases}",
        "",
        f"  {'':16}{'baseline':>10}{'this run':>10}{'delta':>11}",
        f"  {'accuracy':16}{c.accuracy_before:>10.1%}{c.accuracy_after:>10.1%}{c.accuracy_delta_pp:>+9.1f}pp",
        f"  {'pass rate':16}{c.pass_rate_before:>10.1%}{c.pass_rate_after:>10.1%}{c.pass_rate_delta_pp:>+9.1f}pp",
        f"  {'avg relevance':16}{_rel(c.avg_relevance_before):>10}{_rel(c.avg_relevance_after):>10}"
        f"{_signed(c.avg_relevance_delta):>11}",
    ]
    for category, d in c.per_category.items():
        lines.append(
            f"    {category + f' ({d.cases})':14}{d.before:>10.1%}{d.after:>10.1%}{d.delta_pp:>+9.1f}pp"
        )

    lines += ["", f"  Regressions (pass -> fail): {len(c.regressions)}"]
    lines += _changes(c.regressions)
    lines += ["", f"  Improvements (fail -> pass): {len(c.improvements)}"]
    lines += _changes(c.improvements)
    lines += ["", _footer(run), "", f"STATUS: {verdict.status}"]
    lines += [f"  - {reason}" for reason in verdict.reasons]
    return "\n".join(lines)


def format_history(records: list[RunRecord]) -> str:
    if not records:
        return "No runs yet."
    lines = [f"{'run_id':34} {'kind':8} {'prompt':7} {'data':5} {'acc':>6} {'pass':>6} {'rel':>5}  status"]
    for r in records:
        a = r.aggregates
        lines.append(
            f"{r.run_id:34} {r.kind:8} {r.prompt_version:7} v{r.dataset_version:<4} "
            f"{a.accuracy:6.1%} {a.pass_rate:6.1%} {_rel(a.avg_relevance):>5}  {r.status or '-'}"
        )
    return "\n".join(lines)


def _header(r: RunRecord) -> str:
    return (
        f"Run {r.run_id}  [{r.kind}]\n"
        f"  prompt {r.prompt_version} ({r.prompt_hash})  model {r.model}  "
        f"dataset v{r.dataset_version}  {r.aggregates.case_count} cases  git {r.git_sha or 'n/a'}"
    )


def _footer(run: EvalRun) -> str:
    a = run.record.aggregates
    return (
        f"  errors {a.error_count}  |  latency p50 {a.latency_p50_ms:.0f} ms, p95 {a.latency_p95_ms:.0f} ms"
        f"  |  tokens {a.input_tokens} in / {a.output_tokens} out"
        f"  |  cost ${a.cost_usd:.4f} feature + ${a.judge_cost_usd:.4f} judge (peak prices)"
    )


def _changes(changes: list[CaseChange]) -> list[str]:
    lines = [
        f"    {ch.case_id} [{ch.expected_category}, {ch.expected_difficulty}] {ch.reason}"
        for ch in changes[:MAX_LISTED_CHANGES]
    ]
    if len(changes) > MAX_LISTED_CHANGES:
        lines.append(f"    ... and {len(changes) - MAX_LISTED_CHANGES} more")
    return lines


def _rel(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"
