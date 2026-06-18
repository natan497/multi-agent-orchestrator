"""Unit tests for the http_request tool (offline via httpx.MockTransport)."""

import httpx
import pytest

from tools.builtins.http_request import HttpRequest


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_get_returns_status_and_body():
    def handler(request):
        assert request.method == "GET"
        assert request.url.params.get("q") == "hi"
        return httpx.Response(200, text="hello world")

    tool = HttpRequest(client=client_for(handler))
    result = tool.invoke({"url": "https://example.com/api", "params": {"q": "hi"}})
    assert result.ok
    assert "HTTP 200" in result.output
    assert "hello world" in result.output


def test_post_sends_json_body():
    def handler(request):
        assert request.method == "POST"
        import json

        assert json.loads(request.content) == {"a": 1}
        return httpx.Response(201, text="created")

    tool = HttpRequest(client=client_for(handler))
    result = tool.invoke(
        {"url": "https://example.com/items", "method": "POST", "json_body": {"a": 1}}
    )
    assert result.ok
    assert "HTTP 201" in result.output


def test_long_body_is_truncated():
    def handler(request):
        return httpx.Response(200, text="x" * 5000)

    tool = HttpRequest(client=client_for(handler))
    result = tool.invoke({"url": "https://example.com/big"})
    assert "truncated" in result.output
    assert "5000 chars total" in result.output


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",  # non-http scheme
        "http://localhost/admin",  # blocked hostname
        "http://127.0.0.1/secret",  # loopback
        "http://169.254.169.254/meta",  # link-local (cloud metadata)
        "http://10.0.0.5/internal",  # private range
    ],
)
def test_ssrf_and_scheme_guard(url):
    # Guard should trip before any request is attempted.
    tool = HttpRequest(client=client_for(lambda r: httpx.Response(200, text="should not reach")))
    result = tool.invoke({"url": url})
    assert result.ok is False
    assert result.error


def test_connection_error_is_graceful():
    def handler(request):
        raise httpx.ConnectError("boom")

    tool = HttpRequest(client=client_for(handler))
    result = tool.invoke({"url": "https://example.com"})
    assert result.ok is False
    assert "ConnectError" in result.error or "boom" in result.error
