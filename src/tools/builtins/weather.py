"""A real, keyless weather tool backed by Open-Meteo.

Open-Meteo needs no API key. The tool geocodes a place name, then fetches current
conditions. The httpx client is injectable so tests run offline via MockTransport.
"""

from __future__ import annotations

from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from tools.base import Tool

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherArgs(BaseModel):
    location: str = Field(description="A city or place name, e.g. 'Denver' or 'Tokyo, Japan'.")


class Weather(Tool):
    name: ClassVar[str] = "weather"
    description: ClassVar[str] = (
        "Get the current weather (temperature in °C and wind speed) for a city or place "
        "name. Uses the free Open-Meteo API; no key required."
    )
    Args: ClassVar[type[BaseModel]] = WeatherArgs

    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 10.0) -> None:
        self._client = client
        self._timeout = timeout

    def run(self, args: WeatherArgs) -> str:
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            geo = client.get(_GEOCODE_URL, params={"name": args.location, "count": 1})
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                raise ValueError(f"no location found for {args.location!r}")
            place = results[0]
            forecast = client.get(
                _FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,wind_speed_10m",
                },
            )
            forecast.raise_for_status()
            current = forecast.json()["current"]
        finally:
            if self._client is None:
                client.close()

        where = place["name"]
        if place.get("country"):
            where += f", {place['country']}"
        return (
            f"Current weather in {where}: {current['temperature_2m']}°C, "
            f"wind {current['wind_speed_10m']} km/h."
        )
