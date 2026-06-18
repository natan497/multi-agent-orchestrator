"""Runnable demo: drives the orchestrator on a few real tasks and renders the trace.

Requires a GROQ_API_KEY (free tier) in the environment or .env. The tools used here are
keyless (calculator, Open-Meteo weather, Wikipedia, generic HTTP), so the only credential
needed is the Groq key for the planner/executor models.

    python examples/demo_tasks.py
"""

from __future__ import annotations

import os

from rich.console import Console
from rich.panel import Panel

from orchestrator.config import load_config
from orchestrator.models import RunTrace
from orchestrator.orchestrator import Orchestrator
from orchestrator.state import RunResult
from tools.builtins import default_tools
from tools.registry import ToolRegistry

DEMO_TASKS = [
    "What is 1234 * 5678, and is the result greater than five million?",
    "What is the current temperature in Denver, and is it warmer than in Tokyo right now?",
    "Who was Ada Lovelace? Give a one-sentence summary.",
]

_KIND_STYLE = {
    "plan": ("📋", "bold cyan"),
    "replan": ("🔄", "bold yellow"),
    "tool_call": ("🔧", "magenta"),
    "observation": ("👁", "green"),
    "executor_text": ("💬", "blue"),
    "decision": ("⚖", "yellow"),
    "final": ("✅", "bold green"),
    "error": ("❌", "bold red"),
    "limit": ("🛑", "bold red"),
}


def render_trace(console: Console, trace: RunTrace) -> None:
    for event in trace.events:
        icon, style = _KIND_STYLE.get(event.kind, ("•", "white"))
        detail = _format_event(event.kind, event.data)
        console.print(f"  {icon} [{style}]{event.kind}[/]: {detail}")


def _format_event(kind: str, data: dict) -> str:
    if kind in ("plan", "replan"):
        return " → ".join(data.get("steps", [])) or "(no steps)"
    if kind == "tool_call":
        return f"{data.get('name')}({data.get('arguments')})"
    if kind == "observation":
        if data.get("ok"):
            return _truncate(str(data.get("output", "")))
        return f"[red]failed[/]: {data.get('error')}"
    if kind == "decision":
        return str(data.get("decision"))
    return _truncate(str(data))


def _truncate(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def run_task(console: Console, orchestrator: Orchestrator, task: str) -> RunResult:
    console.rule(f"[bold]{task}")
    result = orchestrator.run(task)
    render_trace(console, result.trace)
    answer = result.final_answer or "(no answer)"
    status = "[green]success[/]" if result.success else f"[red]stopped: {result.stop_reason}[/]"
    usage = result.usage
    console.print(
        Panel(
            f"{answer}\n\n"
            f"[dim]{status} · {result.iterations} iterations · {result.tool_calls} tool calls · "
            f"{usage.input_tokens} in / {usage.output_tokens} out tokens "
            f"({usage.cached_tokens} cached)[/]",
            title="Result",
            border_style="green" if result.success else "red",
        )
    )
    return result


def main() -> None:
    console = Console()
    config = load_config()
    if config.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        console.print(
            "[bold red]GROQ_API_KEY is not set.[/] Copy .env.example to .env and add a free "
            "Groq key from https://console.groq.com, then re-run."
        )
        return

    registry = ToolRegistry()
    registry.register_all(default_tools())
    orchestrator = Orchestrator.from_config(config, registry)

    console.print(
        f"[dim]provider={config.provider} · planner={config.planner_model} · "
        f"executor={config.executor_model} · tools={', '.join(registry.names())}[/]\n"
    )
    for task in DEMO_TASKS:
        run_task(console, orchestrator, task)
        console.print()


if __name__ == "__main__":
    main()
