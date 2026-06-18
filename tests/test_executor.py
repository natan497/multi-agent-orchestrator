"""Unit tests for the Executor agent."""

from conftest import ScriptedProvider, text_completion, tool_completion
from orchestrator.executor import Executor
from orchestrator.models import Step, ToolSpec


def test_execute_returns_tool_call():
    provider = ScriptedProvider([tool_completion("calculator", {"expression": "2+2"})])
    ex = Executor(provider)
    result = ex.execute(
        "do math",
        Step(index=0, description="add", tool="calculator"),
        [ToolSpec(name="calculator", description="adds")],
    )
    assert result.called_tool
    assert result.tool_call.name == "calculator"
    assert result.tool_call.arguments == {"expression": "2+2"}
    assert result.usage.total_tokens == 20


def test_execute_passes_tool_specs_to_provider():
    provider = ScriptedProvider([tool_completion("calculator", {"expression": "1"})])
    ex = Executor(provider)
    specs = [ToolSpec(name="calculator", description="adds")]
    ex.execute("g", Step(index=0, description="s"), specs)
    _, sent_tools = provider.calls[0]
    assert sent_tools == specs


def test_execute_returns_text_when_no_tool_called():
    provider = ScriptedProvider([text_completion("no tool needed; the answer is 7")])
    ex = Executor(provider)
    result = ex.execute("g", Step(index=0, description="s"), [])
    assert not result.called_tool
    assert result.text == "no tool needed; the answer is 7"
