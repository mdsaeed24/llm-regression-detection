"""LLM provider implementations and factory."""

from __future__ import annotations

from regress.config import Settings
from regress.providers.base import (
    ChatMessage,
    FatalProviderError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

__all__ = [
    "ChatMessage",
    "FatalProviderError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
    "get_provider",
]


def get_provider(name: str, settings: Settings) -> LLMProvider:
    """Build the provider registered under ``name`` (only DeepSeek today)."""
    if name == "deepseek":
        from regress.providers.deepseek import DeepSeekProvider

        return DeepSeekProvider(settings.provider(name), settings.runtime)
    raise ValueError(f"Unsupported provider '{name}'. Implemented: deepseek")
