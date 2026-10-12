"""Provider-agnostic LLM interface.

Everything above this layer (classifier, judge, runner) depends only on these
types, so adding OpenAI or Anthropic means writing one new class that satisfies
``LLMProvider`` - no changes to the eval pipeline.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class FatalProviderError(RuntimeError):
    """A failure that will repeat for every request (bad key, no balance, unknown model).

    The runner aborts the whole run on this instead of recording 66 identical
    failed cases that would look like a catastrophic prompt regression.
    """


class ChatMessage(BaseModel):
    """One message in a chat completion request."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """A single chat completion request, independent of any vendor SDK."""

    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    json_mode: bool = False
    thinking: bool = False


class TokenUsage(BaseModel):
    """Token counts split the way providers bill them."""

    input_cache_hit: int = 0
    input_cache_miss: int = 0
    output: int = 0

    @property
    def input_total(self) -> int:
        """All input tokens, cached or not."""
        return self.input_cache_hit + self.input_cache_miss

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_cache_hit=self.input_cache_hit + other.input_cache_hit,
            input_cache_miss=self.input_cache_miss + other.input_cache_miss,
            output=self.output + other.output,
        )


class LLMResponse(BaseModel):
    """Normalised completion result."""

    text: str
    model: str
    usage: TokenUsage
    latency_ms: float


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn an ``LLMRequest`` into an ``LLMResponse``."""

    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute one completion, retrying transient failures internally."""
        ...
