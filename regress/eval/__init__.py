"""Evaluation engine: run a prompt over the golden dataset, score, and compare."""

from regress.eval.compare import CaseChange, Comparison, compare_runs
from regress.eval.drift import DriftResult, check_drift
from regress.eval.results import CaseResult, EvalRun, RunAggregates, RunRecord, compute_aggregates
from regress.eval.runner import run_eval
from regress.eval.thresholds import Verdict, apply_drift, evaluate_thresholds

__all__ = [
    "CaseChange",
    "CaseResult",
    "Comparison",
    "DriftResult",
    "EvalRun",
    "RunAggregates",
    "RunRecord",
    "Verdict",
    "apply_drift",
    "check_drift",
    "compare_runs",
    "compute_aggregates",
    "evaluate_thresholds",
    "run_eval",
]
