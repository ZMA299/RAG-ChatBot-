-- Reference schema for Axion RAG Chatbot.
-- Adjust dimensions if you change embedding model.

create extension if not exists vector;

create table if not exists chat_sessions (
  id uuid primary key default gen_random_uuid(),
  title text,
  created_at timestamptz default now()
);

create table if not exists chat_messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references chat_sessions(id) on delete cascade,
  role text not null,
  content text not null,
  latency_s numeric,
  metadata jsonb,
  created_at timestamptz default now()
);

create table if not exists faq_items (
  id uuid primary key default gen_random_uuid(),
  question text not null,
  answer text not null,
  category text,
  embedding vector(1536),
  created_at timestamptz default now()
);

create table if not exists rag_documents (
  id uuid primary key default gen_random_uuid(),
  filename text not null,
  created_at timestamptz default now()
);

create table if not exists rag_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references rag_documents(id) on delete cascade,
  chunk_index integer not null,
  page integer,
  content text not null,
  embedding vector(1536),
  created_at timestamptz default now()
);

create or replace function match_faq_items(
  query_embedding vector(1536),
  match_count int default 3
)
returns table (
  id uuid,
  question text,
  answer text,
  category text,
  similarity float
)
language sql stable
as $$
  select
    faq_items.id,
    faq_items.question,
    faq_items.answer,
    faq_items.category,
    1 - (faq_items.embedding <=> query_embedding) as similarity
  from faq_items
  where faq_items.embedding is not null
  order by faq_items.embedding <=> query_embedding
  limit match_count;
$$;

create or replace function match_rag_chunks(
  query_embedding vector(1536),
  match_count int default 4
)
returns table (
  id uuid,
  document_id uuid,
  chunk_index integer,
  page integer,
  content text,
  similarity float
)
language sql stable
as $$
  select
    rag_chunks.id,
    rag_chunks.document_id,
    rag_chunks.chunk_index,
    rag_chunks.page,
    rag_chunks.content,
    1 - (rag_chunks.embedding <=> query_embedding) as similarity
  from rag_chunks
  where rag_chunks.embedding is not null
  order by rag_chunks.embedding <=> query_embedding
  limit match_count;
$$;
