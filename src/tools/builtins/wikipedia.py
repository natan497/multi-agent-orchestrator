"""A real, keyless Wikipedia search tool.

Uses Wikipedia's REST search endpoint (no key) to return the top match's title,
description, and a plain-text excerpt. The httpx client is injectable for offline tests.
"""

from __future__ import annotations

import re
from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from tools.base import Tool

_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
_TAG_RE = re.compile(r"<[^>]+>")


class WikipediaArgs(BaseModel):
    query: str = Field(description="What to look up on Wikipedia, e.g. 'Alan Turing'.")


class WikipediaSearch(Tool):
    name: ClassVar[str] = "wikipedia_search"
    description: ClassVar[str] = (
        "Search Wikipedia and return the top result's title, short description, and an "
        "excerpt. Use to look up factual information. No key required."
    )
    Args: ClassVar[type[BaseModel]] = WikipediaArgs

    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 10.0) -> None:
        self._client = client
        self._timeout = timeout

    def run(self, args: WikipediaArgs) -> str:
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            resp = client.get(
                _SEARCH_URL,
                params={"q": args.query, "limit": 1},
                headers={"User-Agent": "multi-agent-orchestrator/0.1 (demo)"},
            )
            resp.raise_for_status()
            pages = resp.json().get("pages") or []
        finally:
            if self._client is None:
                client.close()

        if not pages:
            raise ValueError(f"no Wikipedia results for {args.query!r}")
        page = pages[0]
        title = page.get("title", "")
        description = page.get("description") or ""
        excerpt = _TAG_RE.sub("", page.get("excerpt") or "").strip()
        header = f"{title}: {description}" if description else title
        return f"{header}\n{excerpt}".strip()
