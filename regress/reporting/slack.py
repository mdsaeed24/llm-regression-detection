"""Slack alert via an incoming webhook (Block Kit), with a console fallback."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from regress.eval.compare import Comparison
from regress.eval.results import EvalRun
from regress.eval.thresholds import Verdict

logger = logging.getLogger(__name__)

# Emoji + the word itself: status is never carried by color alone.
_STATUS_ICON = {"PASS": ":large_green_circle:", "WARN": ":large_yellow_circle:", "FAIL": ":red_circle:"}
_STATUS_TITLE = {"PASS": "no regression", "WARN": "possible regression", "FAIL": "critical regression"}
MAX_LISTED = 5


def build_payload(
    run: EvalRun, comparison: Comparison, verdict: Verdict, report_location: str
) -> dict[str, Any]:
    """Block Kit message: status header, headline numbers, reasons, top regressions, link."""
    r, c = run.record, comparison
    n = len(c.regressions)
    headline = (
        f"*{n} regression{'s' if n != 1 else ''} detected*, accuracy "
        f"{c.accuracy_before:.1%} → {c.accuracy_after:.1%} ({c.accuracy_delta_pp:+.1f} pp)"
    )
    title = f"{verdict.status}: prompt {r.prompt_version}, {_STATUS_TITLE[verdict.status]}"

    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": f"{_STATUS_ICON[verdict.status]} {title}", "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn", "text": headline}},
        {
            "type": "section",
            "fields": [
                _field("Prompt", f"{r.prompt_version} (`{r.prompt_hash}`)"),
                _field("Model / dataset", f"{r.model} / v{r.dataset_version}"),
                _field("Pass rate", f"{c.pass_rate_before:.1%} → {c.pass_rate_after:.1%}"),
                _field("Avg relevance", f"{_rel(c.avg_relevance_before)} → {_rel(c.avg_relevance_after)}"),
            ],
        },
    ]  # fmt: skip
    if verdict.reasons:
        reasons = "\n".join(f"• {reason}" for reason in verdict.reasons)
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Why*\n{reasons}"}})
    if c.regressions:
        lines = [
            f"• `{ch.case_id}` [{ch.expected_category}, {ch.expected_difficulty}] {ch.reason}"
            for ch in c.regressions[:MAX_LISTED]
        ]
        if n > MAX_LISTED:
            lines.append(f"• …and {n - MAX_LISTED} more in the report")
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Regressed cases*\n" + "\n".join(lines)}})

    link = f"<{report_location}|Open the full report>" if report_location.startswith("http") else f"Report: `{report_location}`"
    blocks.append(
        {"type": "context", "elements": [{"type": "mrkdwn", "text": f"{link} · run `{r.run_id}` · baseline `{c.baseline_run_id}`"}]}
    )  # fmt: skip

    # `text` is the notification/fallback line shown where blocks are not rendered.
    return {"text": f"{title}: {headline.replace('*', '')}", "blocks": blocks}


def send_alert(payload: dict[str, Any], webhook_url: str | None, timeout_s: float) -> bool:
    """POST to Slack, or print when no webhook is configured. Never raises.

    An alerting failure must not change the eval's exit code - the gate is the
    exit code, the alert is a courtesy - so errors are logged and reported as False.
    """
    if not webhook_url:
        print("\n[slack] SLACK_WEBHOOK_URL not set - alert printed instead:")
        print(to_text(payload))
        return False
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - URL from env
            ok = response.status == 200
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("Slack alert failed: %s", exc)
        return False
    if not ok:
        logger.warning("Slack alert failed with HTTP %s", response.status)
    return ok


def to_text(payload: dict[str, Any]) -> str:
    """Plain-text rendering of the Block Kit payload for the console fallback."""
    lines: list[str] = []
    for block in payload["blocks"]:
        if block["type"] in ("header", "section") and "text" in block:
            lines.append(block["text"]["text"])
        if "fields" in block:
            lines += [f["text"].replace("\n", ": ") for f in block["fields"]]
        if block["type"] == "context":
            lines += [e["text"] for e in block["elements"]]
    return "\n".join("  " + line for text in lines for line in text.split("\n"))


def _field(label: str, value: str) -> dict[str, str]:
    return {"type": "mrkdwn", "text": f"*{label}*\n{value}"}


def _rel(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"
