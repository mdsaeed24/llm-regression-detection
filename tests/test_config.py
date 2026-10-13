"""config.yaml loading and REGRESS__SECTION__KEY environment overrides."""

import pytest
from pydantic import ValidationError

from regress.config import load_settings


def test_shipped_config_is_valid():
    s = load_settings("config.yaml")
    assert s.judge.model != "deepseek-flash"  # the judge must not grade its own model
    assert s.thresholds.accuracy_warn_pp <= s.thresholds.accuracy_critical_pp
    s.pricing_for(s.judge.model)  # every model we call has a price


def test_env_overrides_keep_types(monkeypatch):
    monkeypatch.setenv("REGRESS__THRESHOLDS__ACCURACY_CRITICAL_PP", "5")
    monkeypatch.setenv("REGRESS__STORAGE__DB_PATH", "/tmp/x.sqlite")
    monkeypatch.setenv("REGRESS__THRESHOLDS__CATEGORY_CRITICAL_PP", "null")

    s = load_settings("config.yaml")

    assert s.thresholds.accuracy_critical_pp == 5.0
    assert str(s.storage.db_path) == "/tmp/x.sqlite"
    assert s.thresholds.category_critical_pp is None


def test_misspelled_override_is_rejected_not_ignored(monkeypatch):
    monkeypatch.setenv("REGRESS__THRESHOLDS__ACURACY_CRITICAL_PP", "5")

    with pytest.raises(ValidationError, match="acuracy_critical_pp"):
        load_settings("config.yaml")


def test_unknown_model_has_no_silent_zero_price():
    with pytest.raises(KeyError, match="No pricing"):
        load_settings("config.yaml").pricing_for("gpt-unknown")
