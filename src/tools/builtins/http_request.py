"""A generic HTTP request tool (GET/POST) with a basic SSRF guard and a timeout.

This is the "escape hatch" tool: it lets the agent call arbitrary REST endpoints. To keep
it from being trivially abused for SSRF in a portfolio setting, requests are limited to
http/https and obvious internal targets (localhost, link-local, private IP ranges) are
blocked. The httpx client is injectable so tests run fully offline via MockTransport.
"""

from __future__ import annotations

import ipaddress
from typing import ClassVar, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from tools.base import Tool

_MAX_BODY_CHARS = 2000
_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


class HttpRequestArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL to request.")
    method: Literal["GET", "POST"] = "GET"
    params: dict[str, str] | None = Field(default=None, description="Query string parameters.")
    headers: dict[str, str] | None = None
    json_body: dict | None = Field(default=None, description="JSON body for POST requests.")


class HttpRequest(Tool):
    name: ClassVar[str] = "http_request"
    description: ClassVar[str] = (
        "Make an HTTP GET or POST request to a public http(s) URL and return the status "
        "and response body. Use for calling REST APIs."
    )
    Args: ClassVar[type[BaseModel]] = HttpRequestArgs

    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 10.0) -> None:
        self._client = client
        self._timeout = timeout

    def run(self, args: HttpRequestArgs) -> str:
        _guard_url(args.url)
        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=True)
        try:
            resp = client.request(
                args.method,
                args.url,
                params=args.params,
                headers=args.headers,
                json=args.json_body,
            )
        finally:
            if self._client is None:
                client.close()
        body = resp.text
        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + f"\n... [truncated, {len(resp.text)} chars total]"
        return f"HTTP {resp.status_code}\n{body}"


def _guard_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"only http/https URLs are allowed, got scheme {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL has no host: {url!r}")
    if host.lower() in _BLOCKED_HOSTNAMES:
        raise ValueError(f"requests to {host!r} are blocked")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # not a literal IP; allow (hostnames are not resolved here)
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError(f"requests to internal address {host!r} are blocked")
