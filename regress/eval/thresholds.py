"""Turn a baseline comparison into a PASS / WARN / FAIL verdict."""

from __future__ import annotations

from pydantic import BaseModel

from regress.config import ThresholdSettings
from regress.eval.compare import Comparison
from regress.eval.drift import DriftResult
from regress.eval.results import Status

_SEVERITY: dict[Status, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}


class Verdict(BaseModel):
    status: Status
    reasons: list[str]

    @property
    def exit_code(self) -> int:
        """1 blocks the PR; WARN is surfaced but does not block."""
        return 1 if self.status == "FAIL" else 0


def evaluate_thresholds(comparison: Comparison, thresholds: ThresholdSettings) -> Verdict:
    """Apply the thresholds (percentage points, strictly greater than). Worst rule wins."""
    findings: list[tuple[Status, str]] = []

    drop = -comparison.accuracy_delta_pp
    if drop > thresholds.accuracy_critical_pp:
        findings.append(
            ("FAIL", f"accuracy dropped {drop:.1f} pp (critical threshold {thresholds.accuracy_critical_pp:g} pp)")
        )
    elif drop > thresholds.accuracy_warn_pp:
        findings.append(
            ("WARN", f"accuracy dropped {drop:.1f} pp (warning threshold {thresholds.accuracy_warn_pp:g} pp)")
        )

    # The overall number averages over categories, so a collapse in one category
    # (e.g. billing 100% -> 64%) can show up as only a few points overall.
    if thresholds.category_critical_pp is not None:
        for category, delta in comparison.per_category.items():
            category_drop = -delta.delta_pp
            if delta.cases >= thresholds.category_min_cases and category_drop > thresholds.category_critical_pp:
                findings.append(
                    (
                        "FAIL",
                        f"{category} accuracy dropped {category_drop:.1f} pp "
                        f"({delta.before:.0%} -> {delta.after:.0%} over {delta.cases} cases; "
                        f"critical threshold {thresholds.category_critical_pp:g} pp)",
                    )
                )

    if comparison.dataset_mismatch:
        # Shared cases are still compared fairly, but the baseline no longer covers
        # the whole dataset - surface it so someone re-baselines.
        findings.append(
            (
                "WARN",
                f"baseline used dataset v{comparison.baseline_dataset_version}, this run used "
                f"v{comparison.run_dataset_version}; compared {comparison.compared_cases} shared "
                "cases - re-run the baseline",
            )
        )

    status: Status = max((s for s, _ in findings), key=_SEVERITY.__getitem__, default="PASS")
    return Verdict(status=status, reasons=[reason for _, reason in findings])


def apply_drift(verdict: Verdict, drift: DriftResult) -> Verdict:
    """Slow drift is a WARN: a trend worth a human look, not proof this PR broke anything."""
    if not drift.triggered:
        return verdict
    status: Status = verdict.status if verdict.status == "FAIL" else "WARN"
    return Verdict(status=status, reasons=[*verdict.reasons, f"slow drift: {drift.message}"])
