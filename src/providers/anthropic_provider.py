"""Anthropic provider: interface-complete, activated only when a key is present.

This is the "Anthropic-ready" half of the provider-agnostic design. It is never wired
by default — the factory in ``orchestrator.config`` only constructs it when
``ANTHROPIC_API_KEY`` is set and the user selects ``provider="anthropic"``. Groq remains
the default runtime. The translation below targets the Anthropic Messages API; the
``anthropic`` SDK is an optional dependency, imported lazily so the project installs and
runs on Groq alone.
"""

from __future__ import annotations

import os
from typing import Any

from orchestrator.models import Completion, Message, ToolCall, ToolSpec, Usage
from providers.base import LLMProvider, ProviderError


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        client: Any | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(model)
        self._client = client
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._max_tokens = max_tokens
        if self._client is None and not self._api_key:
            raise ProviderError(
                "AnthropicProvider requires ANTHROPIC_API_KEY (or an injected client). "
                "Leave it unset to run on Groq alone."
            )

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # optional dependency
                raise ProviderError(
                    "The 'anthropic' package is not installed. Install it to use the "
                    "Anthropic provider (the project runs on Groq without it)."
                ) from e
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **opts: Any,
    ) -> Completion:
        system, msgs = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": opts.pop("max_tokens", self._max_tokens),
            "messages": msgs,
            **opts,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [self._encode_tool(t) for t in tools]
        raw = self.client.messages.create(**payload)
        return self._parse_completion(raw)

    # ------------------------------------------------------------------ #
    # Translation (Anthropic Messages API)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Anthropic takes the system prompt as a top-level field, not a message."""
        system_parts = [m.content for m in messages if m.role == "system" and m.content]
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content or "",
                            }
                        ],
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                content.extend(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    for tc in m.tool_calls
                )
                msgs.append({"role": "assistant", "content": content})
            else:
                msgs.append({"role": m.role, "content": m.content or ""})
        return ("\n\n".join(system_parts) or None), msgs

    @staticmethod
    def _encode_tool(t: ToolSpec) -> dict[str, Any]:
        return {"name": t.name, "description": t.description, "input_schema": t.parameters}

    @staticmethod
    def _parse_completion(raw: Any) -> Completion:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(raw, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )
        usage_obj = getattr(raw, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
            cached_tokens=getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        )
        return Completion(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=getattr(raw, "stop_reason", None),
            raw=raw,
        )
