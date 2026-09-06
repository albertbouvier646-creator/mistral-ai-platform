"""Long-term memory: embeddings + pgvector similarity search via Supabase."""

import os
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, matches schema.sql


def embed(text: str) -> list[float]:
    return embedder.encode(text, normalize_embeddings=True).tolist()


def save_memory(user_id: str, content: str) -> None:
    supabase.table("memories").insert({
        "user_id": user_id,
        "content": content,
        "embedding": embed(content),
    }).execute()


def search_memories(user_id: str, query: str, top_k: int = 5, threshold: float = 0.5) -> list[str]:
    result = supabase.rpc("match_memories", {
        "query_embedding": embed(query),
        "match_user_id": user_id,
        "match_threshold": threshold,
        "match_count": top_k,
    }).execute()
    return [row["content"] for row in result.data]


def save_message(conversation_id: str, role: str, content: str) -> None:
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }).execute()


def get_conversation_history(conversation_id: str, limit: int = 20) -> list[dict]:
    result = (
        supabase.table("messages")
        .select("role,content")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data
