from __future__ import annotations

from typing import Any

from supabase import create_client

from config import Settings


class SupabaseGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = create_client(settings.supabase_url, settings.supabase_key)

    def create_chat_session(self, title: str | None = None) -> str | None:
        try:
            resp = self.client.table("chat_sessions").insert({"title": title}).execute()
            if resp.data:
                return resp.data[0]["id"]
        except Exception as exc:
            print(f"[Supabase session warning] {type(exc).__name__}: {exc}")
        return None

    def save_chat_message(
        self,
        session_id: str | None,
        role: str,
        content: str,
        latency_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not session_id:
            return

        payload: dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": content,
        }
        if latency_s is not None:
            payload[self.settings.latency_column] = latency_s
        if metadata:
            # Safe optional column. If your table does not have it, the insert will be retried without it.
            payload["metadata"] = metadata

        try:
            self.client.table("chat_messages").insert(payload).execute()
        except Exception as exc:
            if "metadata" in payload:
                payload.pop("metadata", None)
                try:
                    self.client.table("chat_messages").insert(payload).execute()
                    return
                except Exception as retry_exc:
                    print(f"[Supabase chat insert error] {type(retry_exc).__name__}: {retry_exc}")
                    return
            print(f"[Supabase chat insert error] {type(exc).__name__}: {exc}")

    def create_document(self, filename: str) -> str:
        resp = self.client.table("rag_documents").insert({"filename": filename}).execute()
        if not resp.data:
            raise RuntimeError(f"Failed to insert rag_documents row for {filename}: {resp}")
        return resp.data[0]["id"]

    def insert_chunks(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.client.table("rag_chunks").insert(rows).execute()

    def insert_faq_items(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        resp = self.client.table("faq_items").insert(rows).execute()
        return len(resp.data or rows)

    def match_faq_items(self, query_embedding: list[float], match_count: int) -> list[dict[str, Any]]:
        resp = self.client.rpc(
            "match_faq_items",
            {"query_embedding": query_embedding, "match_count": match_count},
        ).execute()
        return resp.data or []

    def match_rag_chunks(self, query_embedding: list[float], match_count: int) -> list[dict[str, Any]]:
        resp = self.client.rpc(
            "match_rag_chunks",
            {"query_embedding": query_embedding, "match_count": match_count},
        ).execute()
        return resp.data or []
