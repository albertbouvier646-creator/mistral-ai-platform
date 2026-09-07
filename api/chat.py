"""
Vercel serverless function: proxies chat requests to the RunPod endpoint.
Keeps the RunPod API key server-side and avoids the browser CORS block that
happens when calling api.runpod.ai directly from a page.
"""

import asyncio
import os
import time

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
PASSCODE = os.environ.get("PASSCODE")  # optional simple shared-link gate
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")  # optional web-search
BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUNPOD_HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

SEARCH_TRIGGERS = [
    "prix", "cours", "actualit", "aujourd'hui", "aujourd’hui", "derni", "récent", "recent",
    "météo", "meteo", "en direct", "résultat", "resultat", "2024", "2025", "2026", "2027",
    "qui est", "qu'est-ce qui", "combien coûte", "combien coute", "quand", "date de sortie",
]


def needs_web_search(message: str) -> bool:
    lowered = message.lower()
    return any(trigger in lowered for trigger in SEARCH_TRIGGERS)


async def search_web(query: str, max_results: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 768
    temperature: float = 0.7


@app.post("/api/chat")
async def chat(req: ChatRequest, x_passcode: str | None = Header(default=None)):
    if PASSCODE and x_passcode != PASSCODE:
        raise HTTPException(status_code=401, detail="Invalid passcode")

    messages = list(req.messages)
    sources: list[dict] = []

    last_user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)
    if TAVILY_API_KEY and last_user_msg and needs_web_search(last_user_msg):
        results = await search_web(last_user_msg)
        if results:
            sources = [{"title": r["title"], "url": r["url"]} for r in results]
            results_block = "\n".join(f"[{i+1}] {r['title']} - {r['url']}\n{r['snippet']}" for i, r in enumerate(results))
            messages = [{"role": "system", "content": (
                "Tu as accès à des résultats de recherche web récents ci-dessous. "
                "Utilise-les pour répondre avec des faits à jour et cite tes sources par numéro [1], [2], etc.\n\n"
                f"Résultats de recherche:\n{results_block}"
            )}] + messages

    async with httpx.AsyncClient(timeout=30) as client:
        submit = await client.post(
            f"{BASE_URL}/run",
            headers=RUNPOD_HEADERS,
            json={"input": {"messages": messages, "max_tokens": req.max_tokens, "temperature": req.temperature, "stream": False}},
        )
        submit.raise_for_status()
        job_id = submit.json()["id"]

        deadline = time.monotonic() + 590  # stay under Vercel's function timeout
        while time.monotonic() < deadline:
            status_resp = await client.get(f"{BASE_URL}/status/{job_id}", headers=RUNPOD_HEADERS)
            status_resp.raise_for_status()
            data = status_resp.json()

            if data["status"] == "COMPLETED":
                content = data["output"][0]["choices"][0]["message"]["content"]
                return {"reply": content, "sources": sources}
            if data["status"] in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise HTTPException(status_code=502, detail=f"RunPod job {data['status']}")

            await asyncio.sleep(3)

    raise HTTPException(status_code=504, detail="Model is still cold-starting, try again in a minute")
