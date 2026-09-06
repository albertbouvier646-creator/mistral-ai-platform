import uuid

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from auth import get_current_user_id
from llm import stream_chat
from memory import get_conversation_history, save_message, search_memories
from prompts import SYSTEM_PROMPT, build_context_messages, needs_web_search
from search import search_web

app = FastAPI(title="Mistral NeMo Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest, user_id: str = Depends(get_current_user_id)):
    conversation_id = req.conversation_id or str(uuid.uuid4())

    history = get_conversation_history(conversation_id)
    memories = search_memories(user_id, req.message)

    search_results = None
    sources = []
    if needs_web_search(req.message):
        search_results = await search_web(req.message)
        sources = [{"title": r["title"], "url": r["url"]} for r in search_results]

    messages = build_context_messages(SYSTEM_PROMPT, memories, history, search_results)
    messages.append({"role": "user", "content": req.message})

    save_message(conversation_id, "user", req.message)

    async def event_stream():
        full_response = ""
        async for token in stream_chat(messages):
            full_response += token
            yield {"event": "token", "data": token}

        save_message(conversation_id, "assistant", full_response)

        if sources:
            yield {"event": "sources", "data": str(sources)}

        yield {"event": "done", "data": conversation_id}

    return EventSourceResponse(event_stream())


@app.get("/health")
async def health():
    return {"status": "ok"}
