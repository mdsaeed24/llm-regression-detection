"""Typed application settings loaded from config.yaml, with secrets from .env."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

DEFAULT_CONFIG_PATH = Path("config.yaml")


class _Strict(BaseModel):
    """Base for config sections: unknown keys are a typo, so reject them."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderSettings(_Strict):
    """Connection settings for one LLM provider."""

    base_url: str
    api_key_env: str
    timeout_s: PositiveFloat = 60.0

    def api_key(self) -> str:
        """Read the API key from the environment, failing with a clear message."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Copy .env.example to .env and add your key."
            )
        return key


class FeatureSettings(_Strict):
    """How the feature under test is called (model/temperature come from the prompt)."""

    provider: str
    thinking: bool = False
    max_tokens: PositiveInt = 300


class JudgeSettings(_Strict):
    """LLM-as-judge model used for summary relevance scoring."""

    provider: str
    model: str
    thinking: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: PositiveInt = 400


class EvaluationSettings(_Strict):
    """What counts as a passing case."""

    relevance_pass_min: float = Field(default=3.0, ge=1.0, le=5.0)


class ThresholdSettings(_Strict):
    """Accuracy drop vs baseline, in percentage points, that triggers WARN / FAIL."""

    accuracy_warn_pp: float = Field(default=3.0, ge=0)
    accuracy_critical_pp: float = Field(default=8.0, ge=0)
    category_critical_pp: float | None = Field(default=20.0, ge=0)
    category_min_cases: PositiveInt = 8

    @model_validator(mode="after")
    def _ordered(self) -> ThresholdSettings:
        if self.accuracy_warn_pp > self.accuracy_critical_pp:
            raise ValueError("accuracy_warn_pp must not exceed accuracy_critical_pp")
        return self


class DriftSettings(_Strict):
    """Moving-average accuracy floor for slow drift."""

    window: PositiveInt = 7
    accuracy_floor: float = Field(default=0.78, ge=0.0, le=1.0)


class ReportingSettings(_Strict):
    """HTML report output."""

    output_dir: Path = Path("reports")
    trend_runs: PositiveInt = 20
    report_url_env: str = "REGRESS_REPORT_URL"


class AlertSettings(_Strict):
    """Slack alerting; falls back to the console when the webhook is unset."""

    slack_webhook_env: str = "SLACK_WEBHOOK_URL"
    timeout_s: PositiveFloat = 10.0

    def webhook_url(self) -> str | None:
        return os.environ.get(self.slack_webhook_env, "").strip() or None


class StorageSettings(_Strict):
    """Where run history is kept."""

    db_path: Path = Path("runs.sqlite")


class ModelPricing(_Strict):
    """USD per 1M tokens (peak rates)."""

    input_cache_hit: float = Field(ge=0)
    input_cache_miss: float = Field(ge=0)
    output: float = Field(ge=0)


class RuntimeSettings(_Strict):
    """Concurrency and retry policy."""

    concurrency: PositiveInt = 5
    max_retries: int = Field(default=4, ge=0)
    backoff_initial_s: PositiveFloat = 1.0
    backoff_max_s: PositiveFloat = 30.0
    validation_retries: int = Field(default=1, ge=0)


class Settings(_Strict):
    """Root settings object; one instance is threaded through the app."""

    providers: dict[str, ProviderSettings]
    feature: FeatureSettings
    judge: JudgeSettings
    pricing: dict[str, ModelPricing]
    runtime: RuntimeSettings = RuntimeSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    thresholds: ThresholdSettings = ThresholdSettings()
    drift: DriftSettings = DriftSettings()
    reporting: ReportingSettings = ReportingSettings()
    alerts: AlertSettings = AlertSettings()
    storage: StorageSettings = StorageSettings()

    def provider(self, name: str) -> ProviderSettings:
        """Return settings for a named provider."""
        try:
            return self.providers[name]
        except KeyError:
            raise KeyError(f"Provider '{name}' is not configured in config.yaml") from None

    def pricing_for(self, model: str) -> ModelPricing:
        """Return pricing for a model, failing loudly so cost is never silently zero."""
        try:
            return self.pricing[model]
        except KeyError:
            raise KeyError(f"No pricing for model '{model}' in config.yaml") from None


def load_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    """Load .env into the process environment, then parse and validate config.yaml."""
    load_dotenv()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Settings.model_validate(raw)
