# Axion RAG Chatbot, Senior Refactor

This version keeps the working behavior of the original bot but separates the system into production-style components:

```text
User
 ↓
Gradio UI / API
 ↓
Intake Agent
- language detection
- small-talk routing
- document-question routing
 ↓
Retrieval Agent
- FAQ semantic search
- PDF vector search
 ↓
Response Agent
- GPT-4o Mini grounded answer generation
 ↓
Grounding Validator
- checks if retrieved context exists
- prevents unsupported answers
 ↓
Supabase Logging
- user message
- assistant message
- latency
- route
- confidence
 ↓
Final Answer
```

## Why there is no medical Safety Agent

The Pharmaceutical Support AI example needs a Safety Agent because medical answers can harm users. This Axion bot is not a medical/pharma assistant, so the medical Safety Agent is replaced by a domain-neutral **Grounding Validator**.

The validator checks whether the answer is supported by retrieved FAQ/PDF context and forces a fallback when the knowledge base does not contain enough information.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio UI, API endpoint, streaming chat orchestration |
| `config.py` | Environment variables and runtime settings |
| `schemas.py` | Shared typed data structures |
| `utils.py` | Language detection, history formatting, confidence helpers |
| `database.py` | Supabase gateway for logs, FAQ search, PDF search |
| `ingestion.py` | PDF and FAQ ingestion pipelines |
| `agents.py` | Intake, retrieval, response, grounding, and orchestration logic |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |
| `supabase_schema.sql` | Optional reference SQL for tables and RPCs |

## Environment Variables

Create `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key
```

## Run

```bash
pip install -r requirements.txt
python app.py
```

## API

The app exposes a Gradio API endpoint:

```http
POST /chat
```

Example response:

```json
{
  "answer": "...",
  "latency": "2.20s (FAQ 0.31s, RAG 0.42s, LLM 1.47s)",
  "sources_pages": [1, 2],
  "confidence": 0.87,
  "route": "pdf_rag",
  "grounded": true
}
```

## Professional Architecture Notes

This architecture is more professional than a single-script RAG bot because it separates responsibilities:

- **Intake Agent** decides how the user message should be handled.
- **Retrieval Agent** owns FAQ and PDF retrieval.
- **Response Agent** owns answer generation.
- **Grounding Validator** prevents unsupported answers.
- **Database Gateway** isolates Supabase calls.
- **Ingestion Service** isolates document processing.

This is still lightweight enough for a student or MVP project, because naturally we are not building a nuclear reactor just to answer company FAQs.
