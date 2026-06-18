"""Unit tests for the Tool base class."""

from typing import ClassVar

from pydantic import BaseModel

from tools.base import Tool


class _GreetArgs(BaseModel):
    name: str
    excited: bool = False


class _Greet(Tool):
    name: ClassVar[str] = "greet"
    description: ClassVar[str] = "Greet someone."
    Args: ClassVar[type[BaseModel]] = _GreetArgs

    def run(self, args: _GreetArgs) -> str:
        return f"Hello {args.name}{'!' if args.excited else ''}"


class _Boom(Tool):
    name: ClassVar[str] = "boom"
    description: ClassVar[str] = "Always fails."
    Args: ClassVar[type[BaseModel]] = _GreetArgs

    def run(self, args: _GreetArgs) -> str:
        raise RuntimeError("kaboom")


def test_to_spec_exposes_json_schema():
    spec = _Greet.to_spec()
    assert spec.name == "greet"
    assert spec.description == "Greet someone."
    assert spec.parameters["properties"]["name"]["type"] == "string"
    assert "title" not in spec.parameters  # stripped for cleaner specs


def test_invoke_validates_and_runs():
    result = _Greet().invoke({"name": "Ada", "excited": True})
    assert result.ok
    assert result.output == "Hello Ada!"


def test_invoke_reports_invalid_arguments_without_raising():
    result = _Greet().invoke({})  # missing required 'name'
    assert result.ok is False
    assert "invalid arguments" in result.error
    assert "name" in result.error


def test_invoke_captures_tool_exception():
    result = _Boom().invoke({"name": "x"})
    assert result.ok is False
    assert "RuntimeError: kaboom" in result.error
