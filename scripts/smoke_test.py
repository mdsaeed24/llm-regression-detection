"""Live smoke test: run every prompt version over a few throwaway emails.

These emails are smoke inputs only - they check that the API wiring, JSON
mode, and validation work end to end. They are NOT evaluation data and must
never be added to the golden dataset.

Usage: python scripts/smoke_test.py [prompts/v1.yaml prompts/v2.yaml ...]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regress.config import load_settings  # noqa: E402
from regress.cost import estimate_cost  # noqa: E402
from regress.feature import EmailClassifier  # noqa: E402
from regress.prompts import load_prompt  # noqa: E402
from regress.providers import TokenUsage, get_provider  # noqa: E402

SMOKE_EMAILS = [
    "The invoice PDF link in my dashboard returns a 404 error, and I need it for my accountant.",
    "yaar password reset mail aa hi nahi raha, spam bhi check kiya. kya karun?",
    "Oh fantastic, ANOTHER price increase with zero notice. Truly loving it.",
]


async def main(prompt_paths: list[str]) -> None:
    settings = load_settings()
    provider = get_provider(settings.feature.provider, settings)

    for path in prompt_paths:
        prompt = load_prompt(path)
        classifier = EmailClassifier(
            prompt, provider, settings.feature, settings.runtime.validation_retries
        )
        outcomes = await asyncio.gather(*(classifier.classify(e) for e in SMOKE_EMAILS))

        total = TokenUsage()
        print(f"\n=== {prompt.version} ({prompt.content_hash}) temp={prompt.temperature} ===")
        for email, outcome in zip(SMOKE_EMAILS, outcomes, strict=True):
            total = total + outcome.usage
            label = outcome.classification.category if outcome.classification else "FAILED"
            summary = outcome.classification.summary if outcome.classification else outcome.error
            print(f"- [{label:9}] {email[:55]}...")
            print(f"    summary : {summary}")
            print(
                f"    {outcome.latency_ms:6.0f} ms | attempts {outcome.attempts} | "
                f"{outcome.usage.input_total} in "
                f"({outcome.usage.input_cache_hit} cached) / {outcome.usage.output} out"
            )
        cost = estimate_cost(total, settings.pricing_for(prompt.model))
        print(f"  total: {total.input_total} in / {total.output} out -> ${cost:.6f} (peak)")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["prompts/v1.yaml", "prompts/v2.yaml"]))
