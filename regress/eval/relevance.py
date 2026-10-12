"""Summary relevance: DeepEval GEval on a 1-5 rubric, scored by the judge model."""

from __future__ import annotations

import os

# DeepEval sends anonymous usage telemetry by default; opt out before importing it.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval.metrics import GEval  # noqa: E402
from deepeval.metrics.g_eval import Rubric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

from regress.config import JudgeSettings  # noqa: E402
from regress.providers.base import ChatMessage, LLMProvider, LLMRequest, TokenUsage  # noqa: E402

# Fixed evaluation steps instead of letting GEval generate them per call: it saves
# an LLM call per case and, more importantly, keeps the judge's criteria identical
# across runs - otherwise a score change could come from the judge, not the prompt.
EVALUATION_STEPS = [
    "Read the Input (a customer support email) and identify the customer's core problem "
    "and what they want done.",
    "Check whether the Actual Output (a one-sentence summary) states that core problem and "
    "the requested action correctly, without facts that are not in the email.",
    "Use the Expected Output (a human-written reference summary) as a guide to what matters. "
    "Do not penalise different wording; if the reference is incomplete, judge against the email.",
    "Penalise summaries that miss the main issue, present a secondary issue as the main one, "
    "or copy personal data such as names, email addresses, or order numbers.",
]

RUBRIC = [
    Rubric(score_range=(1, 1), expected_outcome="Wrong or unrelated: misstates the core problem or invents facts."),
    Rubric(score_range=(2, 2), expected_outcome="Partial: touches the topic but misses the main issue or the request."),
    Rubric(score_range=(3, 3), expected_outcome="Acceptable: core problem right, but vague or missing the request or key specifics."),
    Rubric(score_range=(4, 4), expected_outcome="Good: core problem and request right; minor omissions or wordiness."),
    Rubric(score_range=(5, 5), expected_outcome="Excellent: accurate, complete, specific, concise; as good as the reference."),
]


class RelevanceResult(BaseModel):
    """A judge verdict on one summary."""

    score: float  # 1-5
    reason: str
    usage: TokenUsage


class ProviderJudgeLLM(DeepEvalBaseLLM):
    """Lets DeepEval metrics call the judge through our ``LLMProvider`` layer.

    Used instead of DeepEval's built-in DeepSeek model so that the judge shares
    our retry policy, runs with thinking mode off (so temperature 0 is honoured),
    and reports token usage - judge cost is part of what a run costs. One
    instance per case keeps usage attribution exact under concurrency.
    """

    def __init__(self, provider: LLMProvider, settings: JudgeSettings) -> None:
        self._provider = provider
        self._settings = settings
        self.usage = TokenUsage()
        super().__init__(settings.model)

    def load_model(self) -> LLMProvider:
        return self._provider

    def get_model_name(self) -> str:
        return self._settings.model

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> BaseModel | str:
        response = await self._provider.complete(
            LLMRequest(
                model=self._settings.model,
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                json_mode=True,  # GEval's prompt asks for JSON and includes an example
                thinking=self._settings.thinking,
            )
        )
        self.usage = self.usage + response.usage
        if schema is None:
            return response.text
        try:
            return schema.model_validate_json(response.text)
        except ValidationError:
            return response.text  # DeepEval falls back to its own lenient JSON parsing

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> BaseModel | str:
        raise NotImplementedError("The eval runner is async; use a_measure / a_generate.")


class RelevanceJudge:
    """Scores how well a summary captures an email, against the human reference."""

    def __init__(self, provider: LLMProvider, settings: JudgeSettings) -> None:
        self._provider = provider
        self._settings = settings

    async def score(self, email: str, summary: str, reference: str) -> RelevanceResult:
        llm = ProviderJudgeLLM(self._provider, self._settings)
        metric = GEval(
            name="Summary relevance",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=EVALUATION_STEPS,
            rubric=RUBRIC,
            model=llm,
            async_mode=True,
        )
        await metric.a_measure(
            LLMTestCase(input=email, actual_output=summary, expected_output=reference),
            _show_indicator=False,
        )
        # GEval normalises to [0, 1]; map back onto the rubric's 1-5 scale.
        low, high = metric.score_range
        raw = low + metric.score * (high - low)
        return RelevanceResult(
            score=round(min(max(raw, low), high), 2),
            reason=metric.reason or "",
            usage=llm.usage,
        )
