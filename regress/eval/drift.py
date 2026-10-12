"""Slow drift: a moving average of accuracy across runs, checked against a floor.

Per-run regression checks compare against the *current* baseline. If every PR
loses one point and then becomes the new baseline, no single run ever crosses a
threshold - quality just erodes. The moving average over shipped runs catches it.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel

from regress.config import DriftSettings
from regress.eval.results import RunRecord


class DriftResult(BaseModel):
    window: int
    floor: float
    runs_considered: int
    moving_average: float | None
    triggered: bool
    message: str


def check_drift(history: list[RunRecord], current: RunRecord, settings: DriftSettings) -> DriftResult:
    """Moving average of accuracy over the last ``window`` shipped runs, ``current`` included.

    Only runs on the same dataset version count (accuracy on a different dataset is
    a different measurement), and FAIL runs are excluded because they were blocked
    and never shipped. Needs a full window before it can fire, so a fresh history
    cannot raise a false alarm.
    """
    comparable = [
        r
        for r in [current, *(h for h in history if h.run_id != current.run_id)]
        if r.dataset_version == current.dataset_version and r.status != "FAIL"
    ]
    recent = sorted(comparable, key=lambda r: r.created_at, reverse=True)[: settings.window]

    if len(recent) < settings.window:
        return DriftResult(
            window=settings.window,
            floor=settings.accuracy_floor,
            runs_considered=len(recent),
            moving_average=None,
            triggered=False,
            message=f"not enough history ({len(recent)}/{settings.window} runs on dataset "
            f"v{current.dataset_version})",
        )

    average = statistics.fmean(r.aggregates.accuracy for r in recent)
    triggered = average < settings.accuracy_floor
    verdict = "below" if triggered else "above"
    return DriftResult(
        window=settings.window,
        floor=settings.accuracy_floor,
        runs_considered=len(recent),
        moving_average=average,
        triggered=triggered,
        message=f"{settings.window}-run average accuracy {average:.1%} is {verdict} "
        f"the floor {settings.accuracy_floor:.0%}",
    )
