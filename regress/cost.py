"""Token usage to estimated USD cost."""

from __future__ import annotations

from regress.config import ModelPricing
from regress.providers.base import TokenUsage

_PER = 1_000_000


def estimate_cost(usage: TokenUsage, pricing: ModelPricing) -> float:
    """Estimated cost in USD using peak prices (a conservative upper bound)."""
    return (
        usage.input_cache_hit * pricing.input_cache_hit
        + usage.input_cache_miss * pricing.input_cache_miss
        + usage.output * pricing.output
    ) / _PER
