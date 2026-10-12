"""Prompt YAML loading, validation, hashing, and message rendering."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from regress.prompts import PromptConfig, load_prompt

PROMPTS = sorted(Path("prompts").glob("*.yaml"))


def minimal(**overrides) -> dict:
    data = {
        "version": "vt",
        "created_at": "2026-10-10",
        "model": "deepseek-flash",
        "temperature": 0.0,
        "system_prompt": 'Classify. Reply in json: {"category": "billing", "summary": "..."}',
    }
    return data | overrides


@pytest.mark.parametrize("path", PROMPTS, ids=lambda p: p.name)
def test_every_committed_prompt_loads(path):
    prompt = load_prompt(path)
    assert prompt.version == path.stem
    assert len(prompt.content_hash) == 12


def test_hash_ignores_metadata_but_tracks_behaviour():
    base = PromptConfig.model_validate(minimal())
    assert PromptConfig.model_validate(minimal(description="new words")).content_hash == base.content_hash
    assert PromptConfig.model_validate(minimal(created_at="2027-01-01")).content_hash == base.content_hash
    assert PromptConfig.model_validate(minimal(temperature=0.7)).content_hash != base.content_hash
    assert PromptConfig.model_validate(minimal(model="other")).content_hash != base.content_hash


def test_system_prompt_must_mention_json():
    # DeepSeek's JSON mode rejects prompts that never mention "json".
    with pytest.raises(ValidationError, match="json"):
        PromptConfig.model_validate(minimal(system_prompt="Classify the email."))


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError, match="temprature"):
        PromptConfig.model_validate(minimal(temprature=0.2))


def test_temperature_range():
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(minimal(temperature=3.0))


def test_messages_are_system_then_few_shot_pairs_then_email():
    prompt = PromptConfig.model_validate(
        minimal(few_shot_examples=[{"email": " Refund me ", "output": {"category": "billing", "summary": "S."}}])
    )

    messages = prompt.build_messages("  my app crashes \n")

    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1].content == "Refund me"
    assert yaml.safe_load(messages[2].content) == {"category": "billing", "summary": "S."}
    assert messages[3].content == "my app crashes"  # stripped
