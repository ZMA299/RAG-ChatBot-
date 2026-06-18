from __future__ import annotations

import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    supabase_url: str
    supabase_key: str

    embed_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    transcription_model: str = "gpt-4o-mini-transcribe"

    app_title: str = "Axion RAG Chatbot"
    timezone_name: str = "Asia/Beirut"

    faq_similarity_threshold: float = 0.85
    rag_min_similarity: float = 0.30
    faq_top_k: int = 3
    pdf_top_k: int = 4

    chunk_size: int = 700
    chunk_overlap: int = 120
    max_history_turns: int = 5

    latency_column: str = "latency_s"

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)


def get_settings() -> Settings:
    settings = Settings(
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        supabase_url=(os.getenv("SUPABASE_URL") or "").strip(),
        supabase_key=(os.getenv("SUPABASE_KEY") or "").strip(),
    )

    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.supabase_url:
        missing.append("SUPABASE_URL")
    if not settings.supabase_key:
        missing.append("SUPABASE_KEY")

    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))

    return settings
