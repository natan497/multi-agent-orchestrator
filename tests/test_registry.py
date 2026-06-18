"""Unit tests for the tool registry."""

import pytest

from orchestrator.models import ToolCall
from tools.builtins.calculator import Calculator
from tools.registry import ToolRegistry


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Calculator())
    return reg


def test_register_lookup_and_membership():
    reg = make_registry()
    assert "calculator" in reg
    assert len(reg) == 1
    assert reg.names() == ["calculator"]
    assert isinstance(reg.get("calculator"), Calculator)
    assert reg.get("missing") is None


def test_duplicate_registration_raises():
    reg = make_registry()
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Calculator())


def test_register_all():
    reg = ToolRegistry()
    reg.register_all([Calculator()])
    assert "calculator" in reg


def test_specs_serializes_all_tools():
    reg = make_registry()
    specs = reg.specs()
    assert [s.name for s in specs] == ["calculator"]
    assert specs[0].parameters["properties"]["expression"]["type"] == "string"


def test_dispatch_runs_known_tool():
    reg = make_registry()
    result = reg.dispatch(ToolCall(id="1", name="calculator", arguments={"expression": "2 + 3"}))
    assert result.ok
    assert result.output == "5"


def test_dispatch_unknown_tool_returns_structured_error():
    reg = make_registry()
    result = reg.dispatch(ToolCall(id="1", name="nope", arguments={}))
    assert result.ok is False
    assert "unknown tool 'nope'" in result.error
    assert "calculator" in result.error  # lists what's available


def test_dispatch_validation_error_is_graceful():
    reg = make_registry()
    result = reg.dispatch(ToolCall(id="1", name="calculator", arguments={}))
    assert result.ok is False
    assert "invalid arguments" in result.error
