# Axion RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with:

- OpenAI GPT-4o Mini
- OpenAI Embeddings
- Supabase Vector Database
- Gradio UI
- PDF Knowledge Base
- Voice-to-Text Support

The chatbot retrieves relevant information from uploaded PDF documents stored in Supabase and uses OpenAI models to generate accurate responses.

---

## Features

- PDF document ingestion
- Semantic search using embeddings
- Retrieval-Augmented Generation (RAG)
- Chat history storage in Supabase
- Voice transcription support
- Streaming AI responses
- REST API endpoint
- Arabic and English language support

---

## Architecture

```text
User
 │
 ▼
Gradio UI
 │
 ▼
OpenAI Embeddings
 │
 ▼
Supabase Vector Search
 │
 ▼
Retrieved PDF Chunks
 │
 ▼
GPT-4o Mini
 │
 ▼
Answer
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key
```

Do NOT commit `.env` to GitHub.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AxionBot.git
cd AxionBot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```bash
cp .env.example .env
```

Add your credentials and run:

```bash
python app.py
```

---

## Required Supabase Tables

### chat_sessions

| Column | Type |
|----------|----------|
| id | uuid |
| title | text |

### chat_messages

| Column | Type |
|----------|----------|
| id | uuid |
| session_id | uuid |
| role | text |
| content | text |
| latency_s | numeric |

### rag_documents

| Column | Type |
|----------|----------|
| id | uuid |
| filename | text |

### rag_chunks

| Column | Type |
|----------|----------|
| id | uuid |
| document_id | uuid |
| chunk_index | integer |
| page | integer |
| content | text |
| embedding | vector |

---

## Supabase RPC Function

The application expects a PostgreSQL function named:

```sql
match_rag_chunks
```

This function should perform vector similarity search on the `rag_chunks` table.

---

## API Endpoint

The application exposes:

```http
POST /chat
```

Example:

```json
{
  "message": "What services does Axion provide?"
}
```

Response:

```json
{
  "answer": "...",
  "latency": "1.25s",
  "sources_pages": [1, 2]
}
```

---

## Supported Languages

- English
- Arabic

The chatbot automatically detects the language of the user's message and responds accordingly.

---

## Tech Stack

- Python
- Gradio
- OpenAI GPT-4o Mini
- OpenAI Embeddings
- Supabase
- LangChain
- PyPDFLoader

---

## Security

Never commit:

- `.env`
- OpenAI API keys
- Supabase Service Role keys

Use environment variables and GitHub Secrets when deploying.

---

## License

MIT License
