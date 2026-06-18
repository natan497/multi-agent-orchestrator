"""Groq provider: implements ``LLMProvider`` over Groq's OpenAI-compatible API.

A single instance is constructed with a model id, so the same class backs both the
planner (large model) and the executor (small model). All Groq-specific translation,
error normalization, and rate-limit-aware retry/backoff live here; nothing leaks out.

Rate-limit strategy (see SPEC §5): retries honor HTTP 429 + ``retry-after``; transient
5xx/connection errors back off exponentially with jitter. The binding free-tier
constraint is TPM, so callers keep system prompts stable to benefit from Groq's
automatic prefix caching (cached tokens don't count toward TPM) — this class simply
reports ``cached_tokens`` in :class:`Usage` so that pacing can be measured.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from typing import Any

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

from orchestrator.models import Completion, Message, ToolCall, ToolSpec, Usage
from providers.base import LLMProvider, ProviderError, RateLimitError, ToolFormatError

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Exponential backoff parameters (seconds) used when a 429 carries no retry-after.
_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 30.0


class _RetryableProviderError(ProviderError):
    """Internal marker for transient failures (5xx / connection) worth retrying."""


class GroqProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(model)
        self._client = client
        self._api_key = api_key
        self._max_retries = max_retries
        self._sleep = sleep

    @property
    def client(self) -> Any:
        """Lazily construct the Groq SDK client so import/key aren't needed until used."""
        if self._client is None:
            import groq

            self._client = groq.Groq(
                api_key=self._api_key or os.getenv("GROQ_API_KEY"),
                base_url=GROQ_BASE_URL,
            )
        return self._client

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **opts: Any,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._encode_message(m) for m in messages],
            **opts,
        }
        if tools:
            payload["tools"] = [self._encode_tool(t) for t in tools]
        raw = self._create_with_retry(payload)
        return self._parse_completion(raw)

    # ------------------------------------------------------------------ #
    # Request translation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_message(m: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": m.role}
        # Groq requires content present (may be null for assistant tool-call turns).
        out["content"] = m.content
        if m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            out["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            out["name"] = m.name
        return out

    @staticmethod
    def _encode_tool(t: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }

    # ------------------------------------------------------------------ #
    # Response translation
    # ------------------------------------------------------------------ #
    def _parse_completion(self, raw: Any) -> Completion:
        choice = raw.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError) as e:
                raise ToolFormatError(
                    f"Model returned invalid JSON arguments for tool "
                    f"{tc.function.name!r}: {raw_args!r}"
                ) from e
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return Completion(
            text=getattr(msg, "content", None),
            tool_calls=tool_calls,
            usage=self._parse_usage(getattr(raw, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
            raw=raw,
        )

    @staticmethod
    def _parse_usage(usage: Any) -> Usage:
        if usage is None:
            return Usage()
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details is not None else 0
        return Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=cached,
        )

    # ------------------------------------------------------------------ #
    # Retry / backoff / error normalization
    # ------------------------------------------------------------------ #
    def _create_with_retry(self, payload: dict[str, Any]) -> Any:
        retryer = Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=self._wait,
            retry=retry_if_exception_type((RateLimitError, _RetryableProviderError)),
            sleep=self._sleep,
            reraise=True,
        )
        return retryer(self._create_once, payload)

    def _create_once(self, payload: dict[str, Any]) -> Any:
        try:
            return self.client.chat.completions.create(**payload)
        except Exception as e:  # normalized below; raw SDK errors never escape
            raise self._normalize_error(e) from e

    @staticmethod
    def _wait(retry_state: Any) -> float:
        """Honor ``retry-after`` on 429; otherwise exponential backoff with jitter."""
        exc = retry_state.outcome.exception()
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            return exc.retry_after
        attempt = retry_state.attempt_number
        return min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** (attempt - 1))) + random.uniform(0, 0.25)

    @staticmethod
    def _normalize_error(exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}

        name = type(exc).__name__
        if status == 429 or name == "RateLimitError":
            return RateLimitError(
                str(exc) or "rate limited", retry_after=_parse_retry_after(headers)
            )
        transient_names = {"APIConnectionError", "APITimeoutError", "InternalServerError"}
        if (isinstance(status, int) and status >= 500) or name in transient_names:
            return _RetryableProviderError(str(exc) or "transient provider error")
        return ProviderError(str(exc) or "provider error")


def _parse_retry_after(headers: Any) -> float | None:
    try:
        value = headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
