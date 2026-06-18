"""Tool abstraction.

A ``Tool`` declares a name, a human/LLM-facing description, and a Pydantic ``Args``
model. The args model does double duty: it generates the JSON Schema sent to the model
(via :meth:`Tool.to_spec`) and validates the model's tool-call arguments before
execution (via :meth:`Tool.invoke`). Using Pydantic — already a dependency — keeps the
schema and the validation in one place and avoids a separate JSON-Schema library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from orchestrator.models import ToolSpec


class ToolResult(BaseModel):
    """Outcome of running a tool. The orchestrator maps this onto an Observation."""

    ok: bool
    output: str = ""
    error: str | None = None


class Tool(ABC):
    """Base class for all tools.

    Subclasses set ``name`` and ``description`` and define a nested ``Args`` model, then
    implement :meth:`run`, which receives a validated ``Args`` instance.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]

    @classmethod
    def params_schema(cls) -> dict[str, Any]:
        schema = cls.Args.model_json_schema()
        schema.pop("title", None)
        return schema

    @classmethod
    def to_spec(cls) -> ToolSpec:
        """Provider-neutral spec the registry serializes to each provider's tool format."""
        return ToolSpec(name=cls.name, description=cls.description, parameters=cls.params_schema())

    def invoke(self, raw_args: dict[str, Any]) -> ToolResult:
        """Validate arguments, run the tool, and return a structured result.

        Never raises for ordinary failures: invalid arguments and tool exceptions are
        captured as ``ok=False`` results so a hallucinated argument can't crash a run.
        """
        try:
            args = self.Args.model_validate(raw_args)
        except ValidationError as e:
            return ToolResult(ok=False, error=f"invalid arguments: {_summarize(e)}")
        try:
            output = self.run(args)
        except Exception as e:  # tool-internal failure surfaced to the planner
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        return ToolResult(ok=True, output=output)

    @abstractmethod
    def run(self, args: Any) -> str:
        """Execute the tool with a validated ``Args`` instance and return text output."""
        raise NotImplementedError


def _summarize(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)
