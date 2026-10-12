"""PASS / WARN / FAIL classification, including the per-category rule and drift."""

import pytest

from regress.config import ThresholdSettings
from regress.eval.compare import CategoryDelta, Comparison
from regress.eval.drift import DriftResult
from regress.eval.thresholds import Verdict, apply_drift, evaluate_thresholds

THRESHOLDS = ThresholdSettings(
    accuracy_warn_pp=3, accuracy_critical_pp=8, category_critical_pp=20, category_min_cases=8
)


def comparison(accuracy_delta_pp: float, per_category: dict[str, CategoryDelta] | None = None,
               base_version: int = 1, run_version: int = 1) -> Comparison:  # fmt: skip
    return Comparison(
        baseline_run_id="base", run_id="new",
        baseline_dataset_version=base_version, run_dataset_version=run_version,
        compared_cases=66, only_in_baseline=[], only_in_run=[],
        accuracy_before=0.9, accuracy_after=0.9 + accuracy_delta_pp / 100,
        accuracy_delta_pp=accuracy_delta_pp,
        pass_rate_before=0.8, pass_rate_after=0.8, pass_rate_delta_pp=0.0,
        avg_relevance_before=4.0, avg_relevance_after=4.0, avg_relevance_delta=0.0,
        per_category=per_category or {}, regressions=[], improvements=[],
    )  # fmt: skip


@pytest.mark.parametrize(
    ("delta_pp", "status"),
    [
        (+5.0, "PASS"),  # improvement
        (0.0, "PASS"),
        (-3.0, "PASS"),  # exactly the warn threshold: "exceeds" means strictly greater
        (-3.1, "WARN"),
        (-8.0, "WARN"),  # exactly critical is still only WARN
        (-8.1, "FAIL"),
        (-40.0, "FAIL"),
    ],
)
def test_overall_accuracy_thresholds(delta_pp, status):
    assert evaluate_thresholds(comparison(delta_pp), THRESHOLDS).status == status


def test_category_collapse_fails_even_when_overall_drop_is_small():
    # The v3 scenario: billing 100% -> 64% hides inside a -4.5 pp overall drop.
    per_category = {"billing": CategoryDelta(cases=11, before=1.0, after=0.636, delta_pp=-36.4)}

    verdict = evaluate_thresholds(comparison(-4.5, per_category), THRESHOLDS)

    assert verdict.status == "FAIL"
    assert any("billing accuracy dropped 36.4 pp" in r for r in verdict.reasons)
    assert any("warning threshold" in r for r in verdict.reasons)  # both rules are reported


def test_small_categories_are_ignored_by_the_category_rule():
    # 1 of 4 cases flipping is a 25 pp swing - too few cases to call it a collapse.
    per_category = {"account": CategoryDelta(cases=4, before=1.0, after=0.75, delta_pp=-25.0)}

    assert evaluate_thresholds(comparison(0.0, per_category), THRESHOLDS).status == "PASS"


def test_category_rule_can_be_disabled():
    per_category = {"billing": CategoryDelta(cases=11, before=1.0, after=0.5, delta_pp=-50.0)}
    off = THRESHOLDS.model_copy(update={"category_critical_pp": None})

    assert evaluate_thresholds(comparison(0.0, per_category), off).status == "PASS"


def test_dataset_mismatch_is_at_least_a_warning():
    verdict = evaluate_thresholds(comparison(0.0, base_version=1, run_version=2), THRESHOLDS)

    assert verdict.status == "WARN"
    assert "re-run the baseline" in verdict.reasons[0]


def test_exit_codes():
    assert Verdict(status="PASS", reasons=[]).exit_code == 0
    assert Verdict(status="WARN", reasons=["x"]).exit_code == 0  # surfaced, not blocking
    assert Verdict(status="FAIL", reasons=["x"]).exit_code == 1


def drift(triggered: bool) -> DriftResult:
    return DriftResult(window=7, floor=0.78, runs_considered=7, moving_average=0.75 if triggered else 0.85,
                       triggered=triggered, message="m")  # fmt: skip


@pytest.mark.parametrize(
    ("status", "triggered", "expected"),
    [("PASS", True, "WARN"), ("WARN", True, "WARN"), ("FAIL", True, "FAIL"), ("PASS", False, "PASS")],
)
def test_drift_escalates_to_warn_but_never_downgrades(status, triggered, expected):
    verdict = apply_drift(Verdict(status=status, reasons=[]), drift(triggered))

    assert verdict.status == expected
    assert any(r.startswith("slow drift") for r in verdict.reasons) == triggered


def test_warn_threshold_above_critical_is_rejected():
    with pytest.raises(ValueError, match="must not exceed"):
        ThresholdSettings(accuracy_warn_pp=10, accuracy_critical_pp=5)
