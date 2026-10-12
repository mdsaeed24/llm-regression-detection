"""CLI entry point: ``python -m regress <command>``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from regress.config import Settings, load_settings
from regress.cost import estimate_cost
from regress.dataset import load_dataset
from regress.eval import (
    Comparison,
    DriftResult,
    EvalRun,
    Verdict,
    apply_drift,
    check_drift,
    compare_runs,
    evaluate_thresholds,
    run_eval,
)
from regress.eval.results import RunKind
from regress.feature import EmailClassifier
from regress.prompts import load_prompt
from regress.providers import FatalProviderError, get_provider
from regress.reporting.console import format_comparison, format_history, format_run
from regress.reporting.html import render_report, write_report
from regress.reporting.markdown import format_markdown
from regress.reporting.slack import build_payload, send_alert
from regress.storage import RunStore, export_baseline, import_baseline, load_baseline_file

_HISTORY_SCAN = 500  # recent runs loaded for drift and the trend chart


def _cmd_classify(args: argparse.Namespace) -> int:
    """Classify a single email - a developer tool for poking at a prompt."""
    settings = load_settings(args.config)
    prompt = load_prompt(args.prompt)
    email = Path(args.file).read_text(encoding="utf-8") if args.file else args.email
    if not email:
        print("Provide --email TEXT or --file PATH", file=sys.stderr)
        return 2

    classifier = EmailClassifier(
        prompt,
        get_provider(settings.feature.provider, settings),
        settings.feature,
        settings.runtime.validation_retries,
    )
    outcome = asyncio.run(classifier.classify(email))
    cost = estimate_cost(outcome.usage, settings.pricing_for(prompt.model))

    print(f"prompt   : {prompt.version} ({prompt.content_hash}) model={prompt.model}")
    if outcome.classification:
        print(f"category : {outcome.classification.category}")
        print(f"summary  : {outcome.classification.summary}")
    else:
        print(f"FAILED   : {outcome.error}")
    print(
        f"attempts : {outcome.attempts}  latency: {outcome.latency_ms:.0f} ms  "
        f"tokens: {outcome.usage.input_total} in / {outcome.usage.output} out  "
        f"cost: ${cost:.6f}"
    )
    return 0 if outcome.ok else 1


def _evaluate(args: argparse.Namespace, settings: Settings, kind: RunKind) -> EvalRun:
    """Run the prompt over the dataset with a live progress counter."""
    prompt = load_prompt(args.prompt)
    dataset = load_dataset(args.dataset)
    return asyncio.run(
        run_eval(
            prompt,
            dataset,
            settings,
            feature_provider=get_provider(settings.feature.provider, settings),
            judge_provider=get_provider(settings.judge.provider, settings),
            kind=kind,
            limit=args.limit,
            progress=_progress,
        )
    )


def _progress(done: int, total: int) -> None:
    if sys.stderr.isatty():
        end = "\n" if done == total else ""
        print(f"\r  evaluating {done}/{total}", end=end, file=sys.stderr, flush=True)


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Evaluate a prompt and store the run as the new baseline."""
    settings = load_settings(args.config)
    store = RunStore(settings.storage.db_path)
    run = _evaluate(args, settings, "baseline")
    store.save(run)
    print(format_run(run))
    print(f"\nSaved as the new baseline: {run.record.run_id}")
    path = _write_report(run, store, settings, args.dataset)
    print(f"Report: {path}")
    if args.export:
        exported = export_baseline(run, args.export)
        print(
            f"Exported to {args.export} (+{len(exported.history)} earlier baselines kept for drift). "
            "Commit it to make this the baseline CI compares against."
        )
    return 0


def _resolve_baseline(store: RunStore, spec: str) -> EvalRun | None:
    """'latest' baseline in the DB, a run_id, or a committed baseline .json file."""
    if spec == "latest":
        return store.latest("baseline")
    if spec.endswith(".json"):
        return import_baseline(load_baseline_file(spec), store)
    return store.load(spec)


def _cmd_run(args: argparse.Namespace) -> int:
    """Evaluate a prompt, compare against a baseline, exit 1 on a critical regression."""
    settings = load_settings(args.config)
    store = RunStore(settings.storage.db_path)
    # Resolve the baseline before spending any tokens.
    baseline = _resolve_baseline(store, args.baseline)
    if baseline is None:
        raise LookupError("No baseline yet. Create one with: python -m regress baseline --prompt ...")

    run = _evaluate(args, settings, "run")
    comparison = compare_runs(baseline, run)
    verdict = evaluate_thresholds(comparison, settings.thresholds)
    run = _with_verdict(run, baseline.record.run_id, verdict)
    # Drift is checked after the thresholds: a FAIL run is blocked, so it is
    # excluded from the moving average of what actually ships.
    drift = check_drift(store.list_runs(_HISTORY_SCAN), run.record, settings.drift)
    verdict = apply_drift(verdict, drift)
    run = _with_verdict(run, baseline.record.run_id, verdict)
    store.save(run)

    print(format_comparison(run, comparison, verdict))
    print(f"  drift: {drift.message}")
    path = _write_report(run, store, settings, args.dataset, comparison, verdict, drift)
    print(f"\nReport: {path}")
    location = os.environ.get(settings.reporting.report_url_env) or str(path.resolve())
    if args.summary_md:
        Path(args.summary_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_md).write_text(
            format_markdown(run, comparison, verdict, drift, location), encoding="utf-8"
        )
    if not args.no_alert:
        send_alert(
            build_payload(run, comparison, verdict, location),
            settings.alerts.webhook_url(),
            settings.alerts.timeout_s,
        )
    return verdict.exit_code


def _cmd_report(args: argparse.Namespace) -> int:
    """Re-render the HTML report for a stored run (verdict as recorded at run time)."""
    settings = load_settings(args.config)
    store = RunStore(settings.storage.db_path)
    run = store.latest() if args.run_id == "latest" else store.load(args.run_id)
    if run is None:
        raise LookupError("No runs yet.")
    comparison = drift = None
    if run.record.baseline_run_id:
        comparison = compare_runs(store.load(run.record.baseline_run_id), run)
        earlier = [r for r in store.list_runs(_HISTORY_SCAN) if r.created_at <= run.record.created_at]
        drift = check_drift(earlier, run.record, settings.drift)
    path = _write_report(run, store, settings, args.dataset, comparison, None, drift)
    print(f"Report: {path}")
    return 0


def _with_verdict(run: EvalRun, baseline_run_id: str, verdict: Verdict) -> EvalRun:
    record = run.record.model_copy(
        update={
            "baseline_run_id": baseline_run_id,
            "status": verdict.status,
            "status_reasons": verdict.reasons,
        }
    )
    return run.model_copy(update={"record": record})


def _write_report(
    run: EvalRun,
    store: RunStore,
    settings: Settings,
    dataset_path: str,
    comparison: Comparison | None = None,
    verdict: Verdict | None = None,
    drift: DriftResult | None = None,
) -> Path:
    """Render and write reports/<run_id>.html; history ends at this run."""
    history = [
        r for r in store.list_runs(_HISTORY_SCAN) if r.created_at <= run.record.created_at
    ][: settings.reporting.trend_runs]
    dataset = load_dataset(dataset_path) if Path(dataset_path).exists() else None
    html = render_report(
        run,
        history=list(reversed(history)),
        comparison=comparison,
        verdict=verdict,
        drift=drift,
        dataset=dataset,
    )
    return write_report(html, settings.reporting.output_dir, run.record.run_id)


def _cmd_history(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    print(format_history(RunStore(settings.storage.db_path).list_runs(args.limit)))
    return 0


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", required=True, help="path to a prompt YAML")
    parser.add_argument("--dataset", default="data/golden.json", help="compiled golden dataset")
    parser.add_argument(
        "--limit", type=_positive_int, help="evaluate only the first N cases (by case ID)"
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for tests)."""
    parser = argparse.ArgumentParser(prog="regress", description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    classify = sub.add_parser("classify", help="classify one email with a prompt")
    classify.add_argument("--prompt", required=True, help="path to a prompt YAML")
    group = classify.add_mutually_exclusive_group()
    group.add_argument("--email", help="email text")
    group.add_argument("--file", help="path to a file containing the email")
    classify.set_defaults(func=_cmd_classify)

    baseline = sub.add_parser("baseline", help="evaluate a prompt and store it as the baseline")
    _add_eval_args(baseline)
    baseline.add_argument(
        "--export", metavar="PATH", help="also write the committed baseline file (baselines/baseline.json)"
    )
    baseline.set_defaults(func=_cmd_baseline)

    run = sub.add_parser("run", help="evaluate a prompt and compare it against a baseline")
    _add_eval_args(run)
    run.add_argument(
        "--baseline",
        default="latest",
        help="'latest' (newest baseline in the DB), a run_id, or a baseline .json file",
    )
    run.add_argument("--no-alert", action="store_true", help="skip the Slack/console alert")
    run.add_argument("--summary-md", metavar="PATH", help="write a Markdown summary (PR comment)")
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="re-render the HTML report for a stored run")
    report.add_argument("run_id", nargs="?", default="latest", help="run_id or 'latest'")
    report.add_argument("--dataset", default="data/golden.json", help="for showing email text")
    report.set_defaults(func=_cmd_report)

    history = sub.add_parser("history", help="list recent runs")
    history.add_argument("--limit", type=_positive_int, default=20)
    history.set_defaults(func=_cmd_history)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 ok (PASS/WARN), 1 critical regression (FAIL), 2 setup or infra error."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (FatalProviderError, LookupError, OSError, RuntimeError, ValueError) as exc:
        # Kept distinct from exit 1 so CI can tell "the prompt regressed" from
        # "the eval could not run" (bad key, missing baseline, broken config).
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
