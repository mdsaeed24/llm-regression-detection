"""The public Streamlit demo renders from the committed snapshot (skipped where Streamlit isn't installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]


def test_demo_snapshot_is_committed() -> None:
    assert (ROOT / "demo" / "runs_demo.sqlite").exists()


def test_demo_app_renders_real_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # streamlit_app.py sets these with setdefault; pin them so they are restored after the test.
    monkeypatch.setenv("REGRESS__STORAGE__DB_PATH", "demo/runs_demo.sqlite")
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    monkeypatch.setenv("REGRESS_DEMO", "1")
    app = testing.AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Public demo" in i.value for i in app.info)
    assert not any("No runs yet" in i.value for i in app.info)
