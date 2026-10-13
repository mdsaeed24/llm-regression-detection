"""HTML report, Slack payload, and PR-comment Markdown."""

from datetime import UTC, datetime

from regress.dataset import GoldenCase, GoldenDataset
from regress.eval.compare import compare_runs
from regress.eval.thresholds import Verdict
from regress.reporting.html import render_report
from regress.reporting.markdown import format_markdown
from regress.reporting.slack import build_payload, send_alert, to_text
from tests.conftest import make_result, make_run

HOSTILE = '<script>alert("x")</script> refund | please'


def scenario():
    base = make_run([make_result("a"), make_result("b")], "base", kind="baseline")
    new = make_run([make_result("a", predicted="general"), make_result("b")], "new", minutes=1)
    verdict = Verdict(status="FAIL", reasons=["billing accuracy dropped 50.0 pp"])
    return base, new, compare_runs(base, new), verdict


def test_report_shows_regressions_and_escapes_email_text():
    base, new, comparison, verdict = scenario()
    dataset = GoldenDataset(dataset_version=1, content_hash="h", built_at=datetime.now(UTC), source="t", case_count=1,
                            cases=[GoldenCase(id="a", input_email=HOSTILE, expected_category="billing",
                                              ideal_summary="S.", expected_difficulty="easy")])  # fmt: skip

    html = render_report(new, history=[base.record, new.record], comparison=comparison,
                         verdict=verdict, dataset=dataset)  # fmt: skip

    assert "FAIL" in html and "billing accuracy dropped 50.0 pp" in html
    assert "category billing -&gt; general" in html  # the regression and its reason
    assert "<script>" not in html and "&lt;script&gt;" in html  # emails are untrusted input
    assert "<svg" in html and "http" not in html.split("<body>")[0]  # self-contained: no external assets


def test_baseline_report_renders_without_a_comparison():
    base = make_run([make_result("a")], "base", kind="baseline")

    html = render_report(base, history=[base.record])

    assert "BASELINE" in html and "Scorecard vs baseline" not in html


def test_slack_payload_carries_status_numbers_and_link():
    _, new, comparison, verdict = scenario()

    payload = build_payload(new, comparison, verdict, "https://ci.example/run/1")

    assert payload["blocks"][0]["type"] == "header"
    assert "FAIL" in payload["blocks"][0]["text"]["text"]
    assert "1 regression detected" in payload["text"]
    assert "<https://ci.example/run/1|Open the full report>" in to_text(payload)


def test_alert_without_webhook_prints_and_never_raises(capsys):
    _, new, comparison, verdict = scenario()

    assert send_alert(build_payload(new, comparison, verdict, "reports/x.html"), None, 1) is False
    assert "SLACK_WEBHOOK_URL not set" in capsys.readouterr().out


def test_unreachable_webhook_returns_false_instead_of_failing_the_eval():
    _, new, comparison, verdict = scenario()

    assert send_alert(build_payload(new, comparison, verdict, "x"), "http://127.0.0.1:9/hook", 1) is False


def test_markdown_table_survives_pipes_in_reasons():
    _, new, comparison, verdict = scenario()
    comparison.regressions[0].reason = "error: a | b"

    md = format_markdown(new, comparison, verdict, None, "https://ci.example/run/1")

    assert "### ❌ Prompt `v-test`: **FAIL**" in md
    assert "error: a \\| b" in md
