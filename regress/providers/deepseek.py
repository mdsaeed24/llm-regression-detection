"""DeepSeek provider via the official OpenAI SDK (DeepSeek is OpenAI-compatible)."""

from __future__ import annotations

import logging
import time
from typing import Any

import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from regress.config import ProviderSettings, RuntimeSettings
from regress.providers.base import FatalProviderError, LLMRequest, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)

# Errors worth retrying: rate limits, server errors, network blips. A 400/401
# will fail identically every time, so retrying it only burns time.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)

# Errors that will fail identically for every case: 401 bad key, 402 insufficient
# balance (DeepSeek-specific), 403 forbidden, 404 unknown model.
FATAL_STATUS_CODES = frozenset({401, 402, 403, 404})


class DeepSeekProvider:
    """``LLMProvider`` implementation for the DeepSeek API."""

    name = "deepseek"

    def __init__(
        self,
        settings: ProviderSettings,
        runtime: RuntimeSettings,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._runtime = runtime
        # SDK-level retries are disabled so there is exactly one retry policy
        # (ours), which is configurable and logged.
        self._client = client or AsyncOpenAI(
            api_key=settings.api_key(),
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            max_retries=0,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Call the chat completions endpoint with exponential backoff on transient errors."""
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            stop=stop_after_attempt(self._runtime.max_retries + 1),
            wait=wait_exponential_jitter(
                initial=self._runtime.backoff_initial_s, max=self._runtime.backoff_max_s
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._complete_once(request)
        except openai.APIStatusError as exc:
            if exc.status_code in FATAL_STATUS_CODES:
                raise FatalProviderError(
                    f"DeepSeek returned HTTP {exc.status_code} for model '{request.model}': "
                    f"{exc.message}"
                ) from exc
            raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _complete_once(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            # DeepSeek-specific field, not in the OpenAI SDK signature.
            "extra_body": {"thinking": {"type": "enabled" if request.thinking else "disabled"}},
        }
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        completion = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000

        # DeepSeek's JSON mode can occasionally return empty content; surface it
        # as an empty string so the caller's validation/retry logic handles it.
        text = (completion.choices[0].message.content or "") if completion.choices else ""
        return LLMResponse(
            text=text,
            model=completion.model,
            usage=_parse_usage(completion.usage),
            latency_ms=latency_ms,
        )


def _parse_usage(usage: Any) -> TokenUsage:
    """Map DeepSeek usage (with cache hit/miss extension fields) onto ``TokenUsage``."""
    if usage is None:
        return TokenUsage()
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if hit is None or miss is None:
        # Unknown split: bill everything at the (higher) cache-miss rate.
        hit, miss = 0, prompt_tokens
    return TokenUsage(
        input_cache_hit=int(hit),
        input_cache_miss=int(miss),
        output=int(getattr(usage, "completion_tokens", 0) or 0),
    )
