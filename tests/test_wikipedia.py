"""Unit tests for the wikipedia_search tool (offline via httpx.MockTransport)."""

import httpx

from tools.builtins.wikipedia import WikipediaSearch


def client_returning(payload):
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))


def test_returns_title_description_and_clean_excerpt():
    client = client_returning(
        {
            "pages": [
                {
                    "title": "Ada Lovelace",
                    "description": "English mathematician (1815–1852)",
                    "excerpt": 'Augusta Ada <span class="searchmatch">Lovelace</span> wrote.',
                }
            ]
        }
    )
    result = WikipediaSearch(client=client).invoke({"query": "Ada Lovelace"})
    assert result.ok
    assert "Ada Lovelace: English mathematician" in result.output
    assert "<span" not in result.output  # HTML stripped
    assert "Augusta Ada Lovelace wrote." in result.output


def test_no_results_is_graceful():
    client = client_returning({"pages": []})
    result = WikipediaSearch(client=client).invoke({"query": "asdfqwerzxcv"})
    assert result.ok is False
    assert "no Wikipedia results" in result.error


def test_sends_query_and_user_agent():
    seen = {}

    def handler(request):
        seen["q"] = request.url.params.get("q")
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json={"pages": [{"title": "X", "excerpt": "y"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    WikipediaSearch(client=client).invoke({"query": "quantum computing"})
    assert seen["q"] == "quantum computing"
    assert "multi-agent-orchestrator" in seen["ua"]
