-- Run in Supabase SQL editor. Requires pgvector (Database -> Extensions -> vector).
create extension if not exists vector;

create table if not exists conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) not null,
  title text,
  created_at timestamptz default now()
);

create table if not exists messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid references conversations(id) not null,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  created_at timestamptz default now()
);

create table if not exists memories (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) not null,
  content text not null,
  embedding vector(384) not null,  -- all-MiniLM-L6-v2 dimension
  created_at timestamptz default now()
);

create index if not exists memories_embedding_idx on memories
  using hnsw (embedding vector_cosine_ops);

-- RPC used by backend/memory.py::search_memories
create or replace function match_memories(
  query_embedding vector(384),
  match_user_id uuid,
  match_threshold float,
  match_count int
)
returns table (id uuid, content text, similarity float)
language sql stable
as $$
  select id, content, 1 - (embedding <=> query_embedding) as similarity
  from memories
  where user_id = match_user_id
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

alter table conversations enable row level security;
alter table messages enable row level security;
alter table memories enable row level security;

create policy "users manage own conversations" on conversations
  for all using (auth.uid() = user_id);

create policy "users manage own messages" on messages
  for all using (
    conversation_id in (select id from conversations where user_id = auth.uid())
  );

create policy "users manage own memories" on memories
  for all using (auth.uid() = user_id);
