"""Unit tests for the weather tool (offline via httpx.MockTransport)."""

import httpx

from tools.builtins.weather import Weather


def routing_client(geocode_results, current=None):
    def handler(request):
        if "geocoding-api" in request.url.host:
            return httpx.Response(200, json={"results": geocode_results})
        return httpx.Response(200, json={"current": current or {}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_returns_formatted_current_weather():
    client = routing_client(
        geocode_results=[
            {"name": "Denver", "country": "United States", "latitude": 39.7, "longitude": -105.0}
        ],
        current={"temperature_2m": 22.5, "wind_speed_10m": 8.0},
    )
    result = Weather(client=client).invoke({"location": "Denver"})
    assert result.ok
    assert "Denver, United States" in result.output
    assert "22.5°C" in result.output
    assert "8.0 km/h" in result.output


def test_unknown_location_is_graceful():
    client = routing_client(geocode_results=[])
    result = Weather(client=client).invoke({"location": "Nowheresville"})
    assert result.ok is False
    assert "no location found" in result.error


def test_passes_place_name_to_geocoder():
    seen = {}

    def handler(request):
        if "geocoding-api" in request.url.host:
            seen["name"] = request.url.params.get("name")
            return httpx.Response(
                200,
                json={"results": [{"name": "Tokyo", "latitude": 35.6, "longitude": 139.7}]},
            )
        return httpx.Response(200, json={"current": {"temperature_2m": 18, "wind_speed_10m": 3}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = Weather(client=client).invoke({"location": "Tokyo, Japan"})
    assert result.ok
    assert seen["name"] == "Tokyo, Japan"
    assert "Tokyo" in result.output
