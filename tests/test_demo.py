"""Tests for the demo script's rendering and no-key guard (no network)."""

from examples import demo_tasks
from rich.console import Console

from orchestrator.models import OrchestratorConfig, RunTrace


def test_render_trace_covers_event_kinds():
    trace = RunTrace(goal="g")
    trace.add("plan", steps=["a", "b"])
    trace.add("tool_call", name="calculator", arguments={"expression": "2+2"})
    trace.add("observation", ok=True, output="4")
    trace.add("observation", ok=False, error="boom")
    trace.add("decision", decision="done")

    console = Console(record=True, width=100)
    demo_tasks.render_trace(console, trace)
    out = console.export_text()
    assert "plan" in out and "a → b" in out
    assert "calculator" in out
    assert "boom" in out
    assert "done" in out


def test_main_without_key_prints_guidance(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(demo_tasks, "load_config", lambda: OrchestratorConfig(provider="groq"))
    demo_tasks.main()
    assert "GROQ_API_KEY is not set" in capsys.readouterr().out
