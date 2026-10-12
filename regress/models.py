"""Shared domain models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Category = Literal["billing", "technical", "account", "general"]
CATEGORIES: tuple[Category, ...] = ("billing", "technical", "account", "general")


class Classification(BaseModel):
    """Structured output of the email classifier.

    JSON mode guarantees syntactically valid JSON but not this schema, so every
    model response is validated against this class.
    """

    model_config = ConfigDict(extra="ignore")

    category: Category
    summary: str = Field(min_length=1, max_length=300)

    @field_validator("category", mode="before")
    @classmethod
    def _normalise_category(cls, value: object) -> object:
        """Accept casing/whitespace variants like ' Billing ' - not a real error."""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("summary")
    @classmethod
    def _single_line(cls, value: str) -> str:
        """A one-sentence summary must not span multiple lines."""
        value = value.strip()
        if not value:
            raise ValueError("summary is empty")
        if "\n" in value:
            raise ValueError("summary must be a single sentence on one line")
        return value
