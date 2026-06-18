"""Opt-in live tests that hit real keyless APIs.

Deselected by default (see pyproject `addopts = -m 'not live'`). Run explicitly with:

    pytest -m live

These need network access but no API keys. They're skipped gracefully if the network
is unavailable so a failing connection doesn't look like a code defect.
"""

import httpx
import pytest

from tools.builtins.weather import Weather
from tools.builtins.wikipedia import WikipediaSearch

pytestmark = pytest.mark.live


def test_weather_live():
    try:
        result = Weather().invoke({"location": "Denver"})
    except httpx.HTTPError as e:
        pytest.skip(f"network unavailable: {e}")
    assert result.ok, result.error
    assert "Denver" in result.output
    assert "°C" in result.output


def test_wikipedia_live():
    try:
        result = WikipediaSearch().invoke({"query": "Alan Turing"})
    except httpx.HTTPError as e:
        pytest.skip(f"network unavailable: {e}")
    assert result.ok, result.error
    assert "Turing" in result.output
