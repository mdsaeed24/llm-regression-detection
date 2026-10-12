"""The LLM feature under test: support email -> {category, summary}."""

from __future__ import annotations

import json
import logging
import time

from pydantic import BaseModel, ValidationError

from regress.config import FeatureSettings
from regress.models import Classification
from regress.prompts import PromptConfig
from regress.providers.base import ChatMessage, LLMProvider, LLMRequest, TokenUsage

logger = logging.getLogger(__name__)

_REPAIR_INSTRUCTION = (
    "Your previous reply was not valid for the required format: {error}\n"
    'Reply again with ONLY a json object of the form {{"category": "...", "summary": "..."}}.'
)


class ClassificationOutcome(BaseModel):
    """Everything the eval engine needs from one classification attempt.

    ``classification`` is None when the output never validated; the eval engine
    counts that case as failed rather than dropping it.
    """

    classification: Classification | None
    raw_outputs: list[str]
    usage: TokenUsage
    latency_ms: float
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.classification is not None


class EmailClassifier:
    """Runs a ``PromptConfig`` against a provider and validates the result."""

    def __init__(
        self,
        prompt: PromptConfig,
        provider: LLMProvider,
        feature: FeatureSettings,
        validation_retries: int = 1,
    ) -> None:
        self._prompt = prompt
        self._provider = provider
        self._feature = feature
        self._validation_retries = validation_retries

    async def classify(self, email: str) -> ClassificationOutcome:
        """Classify one email; re-ask once (by default) if validation fails.

        Transient API errors are retried inside the provider. Errors that survive
        those retries propagate - they are infrastructure failures, not model
        quality failures, and the runner decides how to record them.
        """
        messages = self._prompt.build_messages(email)
        raw_outputs: list[str] = []
        usage = TokenUsage()
        error: str | None = None
        started = time.perf_counter()

        for attempt in range(1, self._validation_retries + 2):
            response = await self._provider.complete(
                LLMRequest(
                    model=self._prompt.model,
                    messages=messages,
                    temperature=self._prompt.temperature,
                    max_tokens=self._feature.max_tokens,
                    json_mode=True,
                    thinking=self._feature.thinking,
                )
            )
            raw_outputs.append(response.text)
            usage = usage + response.usage

            try:
                classification = parse_classification(response.text)
            except ValueError as exc:
                error = str(exc)
                logger.debug("Attempt %d failed validation: %s", attempt, error)
                # Show the model its own bad output and the exact error, which
                # fixes far more failures than blindly resending the same request.
                messages = [
                    *messages,
                    ChatMessage(role="assistant", content=response.text or "(empty)"),
                    ChatMessage(role="user", content=_REPAIR_INSTRUCTION.format(error=error)),
                ]
                continue

            return ClassificationOutcome(
                classification=classification,
                raw_outputs=raw_outputs,
                usage=usage,
                latency_ms=_elapsed_ms(started),
                attempts=attempt,
            )

        return ClassificationOutcome(
            classification=None,
            raw_outputs=raw_outputs,
            usage=usage,
            latency_ms=_elapsed_ms(started),
            attempts=len(raw_outputs),
            error=error,
        )


def parse_classification(text: str) -> Classification:
    """Parse model text into a ``Classification`` or raise ``ValueError`` with a short reason."""
    if not text.strip():
        raise ValueError("empty response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json: {exc.msg}") from exc
    try:
        return Classification.model_validate(data)
    except ValidationError as exc:
        reasons = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in exc.errors()
        )
        raise ValueError(reasons) from exc


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
