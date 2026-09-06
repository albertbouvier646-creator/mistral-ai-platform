"""SearXNG web search client."""

import os
import httpx

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")


async def search_web(query: str, language: str = "fr", max_results: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "language": language},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for r in data.get("results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        })
    return results
