"""Self-contained HTML report: one file, inline CSS and SVG, safe to attach to CI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import BaseModel

from regress.dataset import GoldenDataset
from regress.eval.compare import CaseChange, Comparison
from regress.eval.drift import DriftResult
from regress.eval.results import EvalRun, RunRecord
from regress.eval.thresholds import Verdict
from regress.reporting.charts import trend_chart

_env = Environment(
    loader=PackageLoader("regress.reporting", "templates"),
    autoescape=select_autoescape(["html", "j2"]),  # emails are untrusted text
    trim_blocks=True,
    lstrip_blocks=True,
)


class _Tile(BaseModel):
    label: str
    value: str
    before: str = ""
    delta: str = ""
    direction: str = ""


class _ScoreRow(BaseModel):
    label: str
    before: str
    after: str
    delta: str
    direction: str
    sub: bool = False


class _ChangeRow(BaseModel):
    change: CaseChange
    email: str


def render_report(
    run: EvalRun,
    *,
    history: list[RunRecord],
    comparison: Comparison | None = None,
    verdict: Verdict | None = None,
    drift: DriftResult | None = None,
    dataset: GoldenDataset | None = None,
) -> str:
    """Render the report. ``history`` is oldest -> newest and should include ``run``."""
    record, agg = run.record, run.record.aggregates
    emails = {c.id: c.input_email for c in dataset.cases} if dataset else {}

    def rows(changes: list[CaseChange]) -> list[_ChangeRow]:
        return [
            _ChangeRow(change=c, email=emails.get(c.case_id, "(case not in the current dataset)"))
            for c in changes
        ]

    return _env.get_template("report.html.j2").render(
        record=record,
        agg=agg,
        status=verdict.status if verdict else record.status,
        reasons=verdict.reasons if verdict else record.status_reasons,
        comparison=comparison,
        tiles=_tiles(run, comparison),
        scorecard=_scorecard(comparison) if comparison else [],
        regressions=rows(comparison.regressions) if comparison else [],
        improvements=rows(comparison.improvements) if comparison else [],
        history=history,
        drift=drift,
        chart_svg=trend_chart(history, record.run_id, drift.floor if drift else None),
        generated_at=f"{datetime.now(UTC):%Y-%m-%d %H:%M UTC}",
    )


def write_report(html: str, output_dir: Path | str, run_id: str) -> Path:
    path = Path(output_dir) / f"{run_id}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _tiles(run: EvalRun, c: Comparison | None) -> list[_Tile]:
    a = run.record.aggregates
    if c is None:
        return [
            _Tile(label="Accuracy", value=_pct(a.accuracy)),
            _Tile(label="Pass rate", value=_pct(a.pass_rate)),
            _Tile(label="Avg relevance (1-5)", value=_rel(a.avg_relevance)),
            _Tile(label="Cases", value=str(a.case_count)),
        ]
    return [
        _Tile(label="Accuracy", value=_pct(c.accuracy_after), before=_pct(c.accuracy_before),
              delta=_pp(c.accuracy_delta_pp), direction=_direction(c.accuracy_delta_pp)),
        _Tile(label="Pass rate", value=_pct(c.pass_rate_after), before=_pct(c.pass_rate_before),
              delta=_pp(c.pass_rate_delta_pp), direction=_direction(c.pass_rate_delta_pp)),
        _Tile(label="Avg relevance (1-5)", value=_rel(c.avg_relevance_after),
              before=_rel(c.avg_relevance_before), delta=_signed(c.avg_relevance_delta),
              direction=_direction(c.avg_relevance_delta)),
        _Tile(label="Regressions / improvements", value=f"{len(c.regressions)} / {len(c.improvements)}"),
    ]  # fmt: skip


def _scorecard(c: Comparison) -> list[_ScoreRow]:
    rows = [
        _ScoreRow(label="Accuracy", before=_pct(c.accuracy_before), after=_pct(c.accuracy_after),
                  delta=_pp(c.accuracy_delta_pp), direction=_direction(c.accuracy_delta_pp)),
    ]  # fmt: skip
    rows += [
        _ScoreRow(label=f"{category} ({d.cases} cases)", before=_pct(d.before), after=_pct(d.after),
                  delta=_pp(d.delta_pp), direction=_direction(d.delta_pp), sub=True)
        for category, d in c.per_category.items()
    ]  # fmt: skip
    rows += [
        _ScoreRow(label="Pass rate", before=_pct(c.pass_rate_before), after=_pct(c.pass_rate_after),
                  delta=_pp(c.pass_rate_delta_pp), direction=_direction(c.pass_rate_delta_pp)),
        _ScoreRow(label="Avg relevance (1-5)", before=_rel(c.avg_relevance_before),
                  after=_rel(c.avg_relevance_after), delta=_signed(c.avg_relevance_delta),
                  direction=_direction(c.avg_relevance_delta)),
    ]  # fmt: skip
    return rows


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _pp(v: float) -> str:
    return f"{v:+.1f} pp".replace("-", "−")


def _rel(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _signed(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}".replace("-", "−")


def _direction(delta: float | None) -> str:
    """CSS class for a delta; tiny changes stay neutral ink rather than shouting."""
    if delta is None or abs(delta) < 0.05:
        return ""
    return "up" if delta > 0 else "down"
