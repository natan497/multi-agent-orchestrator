"""Tool registry: registration, lookup, spec serialization, and safe dispatch."""

from __future__ import annotations

from orchestrator.models import ToolCall, ToolSpec
from tools.base import Tool, ToolResult


class ToolRegistry:
    """Holds the tools available to a run and dispatches model tool-calls to them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Provider-neutral specs for every registered tool, ready to send to a model."""
        return [tool.to_spec() for tool in self._tools.values()]

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Run the tool named by ``call``. Unknown names return a structured error
        (fed back to the planner) rather than raising."""
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names()) or "(none)"
            return ToolResult(
                ok=False,
                error=f"unknown tool {call.name!r}. available tools: {available}",
            )
        return tool.invoke(call.arguments)
