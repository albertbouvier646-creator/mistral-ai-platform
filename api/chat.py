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
BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUNPOD_HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

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

    async with httpx.AsyncClient(timeout=30) as client:
        submit = await client.post(
            f"{BASE_URL}/run",
            headers=RUNPOD_HEADERS,
            json={"input": {"messages": req.messages, "max_tokens": req.max_tokens, "temperature": req.temperature, "stream": False}},
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
                return {"reply": content}
            if data["status"] in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise HTTPException(status_code=502, detail=f"RunPod job {data['status']}")

            await asyncio.sleep(3)

    raise HTTPException(status_code=504, detail="Model is still cold-starting, try again in a minute")
