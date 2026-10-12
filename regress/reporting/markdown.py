"""Markdown summary for the PR comment and the GitHub Actions job summary."""

from __future__ import annotations

from regress.eval.compare import Comparison
from regress.eval.drift import DriftResult
from regress.eval.results import EvalRun
from regress.eval.thresholds import Verdict

_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
MAX_LISTED = 10


def format_markdown(
    run: EvalRun,
    comparison: Comparison,
    verdict: Verdict,
    drift: DriftResult | None,
    report_location: str,
) -> str:
    r, c = run.record, comparison
    lines = [
        f"### {_ICON[verdict.status]} Prompt `{r.prompt_version}`: **{verdict.status}**",
        "",
        f"{len(c.regressions)} regression(s), {len(c.improvements)} improvement(s) on "
        f"{c.compared_cases} golden cases · model `{r.model}` · dataset v{r.dataset_version} · "
        f"baseline `{c.baseline_run_id}`",
        "",
    ]
    if verdict.reasons:
        lines += [f"- {reason}" for reason in verdict.reasons] + [""]

    lines += [
        "| Metric | Baseline | This PR | Change |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {c.accuracy_before:.1%} | {c.accuracy_after:.1%} | {c.accuracy_delta_pp:+.1f} pp |",
    ]
    lines += [
        f"| &nbsp;&nbsp;{category} ({d.cases}) | {d.before:.1%} | {d.after:.1%} | {d.delta_pp:+.1f} pp |"
        for category, d in c.per_category.items()
    ]
    lines += [
        f"| Pass rate | {c.pass_rate_before:.1%} | {c.pass_rate_after:.1%} | {c.pass_rate_delta_pp:+.1f} pp |",
        f"| Avg relevance (1-5) | {_num(c.avg_relevance_before)} | {_num(c.avg_relevance_after)} | "
        f"{_signed(c.avg_relevance_delta)} |",
        "",
    ]

    if c.regressions:
        lines += [
            f"<details><summary>Regressed cases ({len(c.regressions)})</summary>",
            "",
            "| Case | Expected | Why |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{ch.case_id}` | {ch.expected_category} · {ch.expected_difficulty} | {_cell(ch.reason)} |"
            for ch in c.regressions[:MAX_LISTED]
        ]
        if len(c.regressions) > MAX_LISTED:
            lines.append(f"| … | | {len(c.regressions) - MAX_LISTED} more in the report |")
        lines += ["", "</details>", ""]

    a = r.aggregates
    footer = (
        f"Full report (old vs new output side by side): {report_location} · "
        f"cost ${a.cost_usd + a.judge_cost_usd:.4f} · run `{r.run_id}`"
    )
    if drift:
        footer += f" · drift: {drift.message}"
    lines.append(f"<sub>{footer}</sub>")
    return "\n".join(lines) + "\n"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _signed(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}"


def _cell(text: str) -> str:
    """Keep table cells on one row: pipes and newlines would break the Markdown table."""
    return text.replace("|", "\\|").replace("\n", " ")
