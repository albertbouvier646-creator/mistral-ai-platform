"""Client for the RunPod Serverless endpoint running the GGUF model."""

import json
import os
from typing import AsyncGenerator

import httpx

RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}


async def stream_chat(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> AsyncGenerator[str, None]:
    """Streams tokens from RunPod's /run + /stream job API."""
    payload = {
        "input": {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
    }

    async with httpx.AsyncClient(timeout=120) as client:
        submit = await client.post(f"{BASE_URL}/run", headers=HEADERS, json=payload)
        submit.raise_for_status()
        job_id = submit.json()["id"]

        seen = 0
        while True:
            resp = await client.get(f"{BASE_URL}/stream/{job_id}", headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("stream", [])[seen:]:
                yield item["output"]
            seen = len(data.get("stream", []))

            if data.get("status") in ("COMPLETED", "FAILED"):
                if data.get("status") == "FAILED":
                    raise RuntimeError(f"RunPod job failed: {data}")
                break
