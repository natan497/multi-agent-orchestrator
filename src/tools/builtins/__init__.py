"""Built-in tools shipped with the orchestrator."""

from __future__ import annotations

from tools.base import Tool
from tools.builtins.calculator import Calculator
from tools.builtins.http_request import HttpRequest
from tools.builtins.weather import Weather
from tools.builtins.wikipedia import WikipediaSearch

__all__ = ["Calculator", "HttpRequest", "Weather", "WikipediaSearch", "default_tools"]


def default_tools() -> list[Tool]:
    """Fresh instances of every built-in tool, ready to register."""
    return [Calculator(), HttpRequest(), Weather(), WikipediaSearch()]
