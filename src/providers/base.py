"""Provider-agnostic LLM interface.

Every concrete provider (Groq today, Anthropic-ready) implements ``LLMProvider`` and
translates to/from its own wire format internally. The orchestrator depends only on
this interface and the normalized models in ``orchestrator.models`` — it never imports
a provider SDK or a concrete provider directly.

Errors are normalized too: providers raise the exceptions defined here, never raw SDK
exceptions, so retry/backoff and the orchestrator can reason about failures uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from orchestrator.models import Completion, Message, ToolSpec


class ProviderError(Exception):
    """Base class for all provider failures surfaced to the orchestrator."""


class RateLimitError(ProviderError):
    """Raised on HTTP 429 / quota exhaustion.

    ``retry_after`` (seconds), when present, comes from the ``retry-after`` header and
    lets the backoff layer wait precisely instead of guessing.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ToolFormatError(ProviderError):
    """Raised when a tool spec or a model's tool-call payload cannot be translated."""


class LLMProvider(ABC):
    """Normalized chat-completion interface with tool calling.

    Concrete providers are constructed with a model id, so a single class can back both
    the planner (large model) and the executor (small model) under different config.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def name(self) -> str:
        """Short provider name (e.g. ``"groq"``, ``"anthropic"``) for traces/logs."""
        return type(self).__name__.removesuffix("Provider").lower()

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **opts: Any,
    ) -> Completion:
        """Run one chat completion.

        Args:
            messages: Conversation in normalized form.
            tools: Provider-neutral tool specs the model may call, or ``None``.
            **opts: Provider-tunable knobs (e.g. ``temperature``, ``max_tokens``).

        Returns:
            A normalized :class:`~orchestrator.models.Completion`.

        Raises:
            RateLimitError: on 429 / quota exhaustion.
            ToolFormatError: if tools or tool-call payloads cannot be translated.
            ProviderError: for any other provider-side failure.
        """
        raise NotImplementedError
