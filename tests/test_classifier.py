"""The feature under test with a mocked LLM: parsing, validation, and the repair retry."""

import pytest

from regress.config import FeatureSettings
from regress.feature import EmailClassifier, parse_classification
from regress.prompts import load_prompt
from tests.conftest import FakeProvider, classification_json

FEATURE = FeatureSettings(provider="fake", thinking=False, max_tokens=300)


def classifier(provider: FakeProvider, retries: int = 1) -> EmailClassifier:
    return EmailClassifier(load_prompt("prompts/v1.yaml"), provider, FEATURE, retries)


async def test_valid_output_on_first_attempt():
    provider = FakeProvider(lambda req: classification_json(" Billing "))

    outcome = await classifier(provider).classify("charged twice")

    assert outcome.ok and outcome.attempts == 1
    assert outcome.classification.category == "billing"  # case/whitespace normalised
    request = provider.requests[0]
    assert request.json_mode and not request.thinking and request.temperature == 0.0


async def test_invalid_output_is_repaired_with_the_error_shown_to_the_model():
    replies = iter(['{"category": "shipping", "summary": "x"}', classification_json("technical")])
    provider = FakeProvider(lambda req: next(replies))

    outcome = await classifier(provider).classify("app crashes")

    assert outcome.ok and outcome.attempts == 2
    assert outcome.usage.output == 40  # usage summed across both attempts
    repair = provider.requests[1].messages
    assert repair[-2].role == "assistant" and "shipping" in repair[-2].content
    assert "category" in repair[-1].content and "json" in repair[-1].content


async def test_unrecoverable_output_is_a_failed_case_not_an_exception():
    provider = FakeProvider(lambda req: "not json at all")

    outcome = await classifier(provider, retries=1).classify("hello")

    assert not outcome.ok
    assert outcome.attempts == 2
    assert "invalid json" in outcome.error
    assert outcome.raw_outputs == ["not json at all"] * 2


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty response"),
        ("{", "invalid json"),
        ('{"category": "billing"}', "summary"),
        ('{"category": "billing", "summary": "line one\\nline two"}', "single sentence"),
        ('{"category": "refunds", "summary": "x"}', "category"),
    ],
)
def test_parse_classification_explains_failures(text, reason):
    with pytest.raises(ValueError, match=reason):
        parse_classification(text)
