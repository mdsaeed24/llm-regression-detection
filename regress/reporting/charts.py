"""Inline SVG charts for the self-contained HTML report (no JS libraries, no CDN).

Colors are CSS custom properties defined by the report template, so the same
SVG follows the page's light/dark theme.
"""

from __future__ import annotations

import math
from html import escape

from regress.eval.results import RunRecord

WIDTH, HEIGHT = 720, 260
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 44, 64, 16, 36
SERIES = (  # (label, accessor, CSS variable) - fixed order, never cycled
    ("Accuracy", lambda r: r.aggregates.accuracy, "--series-1"),
    ("Pass rate", lambda r: r.aggregates.pass_rate, "--series-2"),
)
MIN_LABEL_GAP_PX = 14


def trend_chart(runs: list[RunRecord], current_run_id: str, drift_floor: float | None) -> str:
    """Accuracy and pass rate over runs (oldest -> newest), one shared 0-100% axis."""
    if not runs:
        return '<p class="muted">No run history yet.</p>'

    values = [fn(r) for r in runs for _, fn, _ in SERIES]
    if drift_floor is not None:
        values.append(drift_floor)
    # Zoom the y-axis to the data (a line chart need not start at zero), on clean 10% steps.
    y_min = max(0.0, math.floor(min(values) * 10 - 0.5) / 10)
    y_max = 1.0

    plot_w = WIDTH - PAD_LEFT - PAD_RIGHT
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM

    def x(i: int) -> float:
        return PAD_LEFT + (plot_w / 2 if len(runs) == 1 else i * plot_w / (len(runs) - 1))

    def y(v: float) -> float:
        return PAD_TOP + (y_max - v) / (y_max - y_min) * plot_h

    parts = [
        f'<svg class="chart" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="Accuracy and pass rate over the last {len(runs)} runs">'
    ]

    # Gridlines + y ticks (recessive hairlines).
    steps = round((y_max - y_min) * 10)
    for k in range(steps + 1):
        v = y_min + k / 10
        parts.append(
            f'<line class="grid" x1="{PAD_LEFT}" x2="{WIDTH - PAD_RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
            f'<text class="tick" x="{PAD_LEFT - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>'
        )

    # Drift floor: a labelled reference line, not a data series.
    if drift_floor is not None:
        parts.append(
            f'<line class="ref" x1="{PAD_LEFT}" x2="{WIDTH - PAD_RIGHT}" '
            f'y1="{y(drift_floor):.1f}" y2="{y(drift_floor):.1f}"/>'
            # Labelled at the left edge: the right edge belongs to the series end labels.
            f'<text class="tick" x="{PAD_LEFT + 6}" y="{y(drift_floor) - 6:.1f}">'
            f"drift floor {drift_floor:.0%}</text>"
        )

    # X labels: prompt version under each run; the current run is emphasised.
    for i, r in enumerate(runs):
        cls = "tick current" if r.run_id == current_run_id else "tick"
        parts.append(
            f'<text class="{cls}" x="{x(i):.1f}" y="{HEIGHT - PAD_BOTTOM + 18}" '
            f'text-anchor="middle">{escape(r.prompt_version)}</text>'
        )

    # Lines first, then markers on top (markers carry a surface ring).
    end_labels: list[tuple[float, str]] = []
    for _, fn, var in SERIES:
        points = " ".join(f"{x(i):.1f},{y(fn(r)):.1f}" for i, r in enumerate(runs))
        parts.append(f'<polyline class="line" style="stroke: var({var})" points="{points}"/>')
    for _, fn, var in SERIES:
        for i, r in enumerate(runs):
            radius = 5 if r.run_id == current_run_id else 4
            parts.append(
                f'<circle class="dot" style="fill: var({var})" cx="{x(i):.1f}" cy="{y(fn(r)):.1f}" r="{radius}"/>'
            )
        last_y = y(fn(runs[-1]))
        end_labels.append((last_y, f"{fn(runs[-1]):.1%}"))

    # Direct end labels, skipped where they would collide (the legend still names both).
    placed: list[float] = []
    for label_y, text in end_labels:
        if all(abs(label_y - p) >= MIN_LABEL_GAP_PX for p in placed):
            parts.append(
                f'<text class="value" x="{x(len(runs) - 1) + 10:.1f}" y="{label_y + 4:.1f}">{text}</text>'
            )
            placed.append(label_y)

    # Hover layer: one generous invisible hit target per run with a native tooltip.
    for i, r in enumerate(runs):
        a = r.aggregates
        tip = (
            f"{r.run_id}\nprompt {r.prompt_version} · {r.kind}"
            f"{' · ' + r.status if r.status else ''}\n"
            f"accuracy {a.accuracy:.1%} · pass rate {a.pass_rate:.1%}"
        )
        band = plot_w / max(1, len(runs) - 1)
        parts.append(
            f'<rect class="hit" x="{x(i) - band / 2:.1f}" y="{PAD_TOP}" width="{band:.1f}" '
            f'height="{plot_h}"><title>{escape(tip)}</title></rect>'
        )

    parts.append("</svg>")
    return "".join(parts)
