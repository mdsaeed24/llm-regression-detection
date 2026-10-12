"""Versioned prompt configurations loaded from prompts/*.yaml."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from regress.models import Classification
from regress.providers.base import ChatMessage


class FewShotExample(BaseModel):
    """A demonstration pair rendered as a user/assistant turn."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=1)
    output: Classification


class PromptConfig(BaseModel):
    """One immutable prompt version: the unit that gets regression-tested."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    created_at: date | datetime
    description: str = ""
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    system_prompt: str = Field(min_length=1)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    @field_validator("system_prompt")
    @classmethod
    def _mentions_json(cls, value: str) -> str:
        """DeepSeek JSON mode rejects prompts that never mention 'json'."""
        if "json" not in value.lower():
            raise ValueError(
                "system_prompt must contain the word 'json' and an example of the "
                "expected format (DeepSeek JSON Output requirement)"
            )
        return value

    @property
    def content_hash(self) -> str:
        """Short hash of everything that affects model behaviour.

        Two files with the same ``version`` but different content are a bug;
        storing this hash with each run makes that detectable.
        """
        payload = self.model_dump(mode="json", exclude={"created_at", "description"})
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return digest[:12]

    def build_messages(self, email: str) -> list[ChatMessage]:
        """Render system prompt, few-shot turns, and the email under test."""
        messages = [ChatMessage(role="system", content=self.system_prompt)]
        for example in self.few_shot_examples:
            messages.append(ChatMessage(role="user", content=example.email.strip()))
            messages.append(
                ChatMessage(role="assistant", content=example.output.model_dump_json())
            )
        messages.append(ChatMessage(role="user", content=email.strip()))
        return messages


def load_prompt(path: Path | str) -> PromptConfig:
    """Parse and validate a prompt YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return PromptConfig.model_validate(raw)
