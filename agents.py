from __future__ import annotations

import time
from typing import Any, Generator

from openai import OpenAI

from config import Settings
from database import SupabaseGateway
from schemas import IntakeResult, RetrievalHit, RAGResult
from utils import (
    confidence_from_hits,
    detect_lang,
    format_history,
    is_smalltalk,
    normalize_text,
    now_string,
)


class IntakeAgent:
    """Fast deterministic router. Keep this boring. Boring systems break less."""

    def classify(self, question: str) -> IntakeResult:
        normalized = normalize_text(question)
        language = detect_lang(question)

        if not normalized:
            intent = "out_of_scope"
        elif is_smalltalk(normalized):
            intent = "greeting"
        else:
            intent = "document_question"

        return IntakeResult(
            language=language,  # type: ignore[arg-type]
            intent=intent,  # type: ignore[arg-type]
            normalized_question=normalized,
            should_retrieve=intent == "document_question",
        )


class RetrievalAgent:
    def __init__(self, settings: Settings, db: SupabaseGateway, embeddings: Any):
        self.settings = settings
        self.db = db
        self.embeddings = embeddings

    def retrieve_faq(self, question: str) -> list[dict[str, Any]]:
        q_vec = self.embeddings.embed_query(question)
        return self.db.match_faq_items(q_vec, self.settings.faq_top_k)

    def retrieve_pdf(self, question: str) -> list[dict[str, Any]]:
        q_vec = self.embeddings.embed_query(question)
        return self.db.match_rag_chunks(q_vec, self.settings.pdf_top_k)

    def best_faq_answer(self, question: str) -> tuple[str | None, dict[str, Any] | None]:
        hits = self.retrieve_faq(question)
        if not hits:
            return None, None

        best = hits[0]
        similarity = float(best.get("similarity") or 0)
        if similarity >= self.settings.faq_similarity_threshold:
            return (best.get("answer") or "").strip(), best

        return None, best

    def to_retrieval_hits(self, hits: list[dict[str, Any]], source_type: str) -> list[RetrievalHit]:
        converted: list[RetrievalHit] = []
        for hit in hits:
            converted.append(
                RetrievalHit(
                    source_type=source_type,  # type: ignore[arg-type]
                    content=hit.get("content") or hit.get("answer") or "",
                    page=hit.get("page"),
                    similarity=float(hit.get("similarity") or 0) if hit.get("similarity") is not None else None,
                    metadata=hit,
                )
            )
        return converted


class GroundingValidator:
    """Domain-neutral guardrail replacing the medical Safety Agent.

    It checks whether the answer is supported by retrieved context. It does not pretend
    to do clinical safety review, because this bot is not a pharmaceutical assistant.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def has_reliable_context(self, hits: list[dict[str, Any]]) -> bool:
        if not hits:
            return False
        similarities = [hit.get("similarity") for hit in hits]
        confidence = confidence_from_hits(similarities)
        return confidence >= self.settings.rag_min_similarity

    def fallback(self, lang: str) -> str:
        if lang == "ar":
            return "لا أملك معلومات كافية من قاعدة المعرفة للإجابة على هذا السؤال بدقة."
        return "I do not have enough information in the knowledge base to answer this accurately."


class ResponseAgent:
    def __init__(self, settings: Settings, openai_client: OpenAI):
        self.settings = settings
        self.client = openai_client

    def system_prompt(self, lang: str) -> str:
        return f"""You are an AI Receptionist and Customer Support Assistant for Axion.

ROLE:
- Answer customer questions using only the supplied FAQ or PDF knowledge-base context.
- Answer in the user's language: {'Arabic' if lang == 'ar' else 'English'}.

GROUNDING RULES:
- Do not invent company details, policies, services, prices, names, phone numbers, emails, or deadlines.
- If the context does not contain the answer, say you do not have enough information.
- Keep the answer concise, professional, and human-readable.
- If PDF pages are provided in the context, cite the relevant page numbers in the final answer.

Current date/time: {now_string(self.settings)}
"""

    def build_user_prompt(
        self,
        question: str,
        history: list[tuple[str, str]] | None,
        pdf_context: str,
    ) -> str:
        return (
            f"Conversation so far:\n{format_history(history, self.settings.max_history_turns)}\n\n"
            f"Knowledge-base context:\n{pdf_context or '(none)'}\n\n"
            f"User question: {question}"
        )

    def stream_answer(
        self,
        question: str,
        history: list[tuple[str, str]] | None,
        lang: str,
        pdf_context: str,
    ) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": self.system_prompt(lang)},
            {"role": "user", "content": self.build_user_prompt(question, history, pdf_context)},
        ]

        stream = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=0,
            stream=True,
        )

        full = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full += token
                yield full.strip()


class RAGOrchestrator:
    def __init__(
        self,
        settings: Settings,
        intake: IntakeAgent,
        retrieval: RetrievalAgent,
        response: ResponseAgent,
        validator: GroundingValidator,
    ):
        self.settings = settings
        self.intake = intake
        self.retrieval = retrieval
        self.response = response
        self.validator = validator

    def answer_stream(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> Generator[tuple[str, str, list[dict[str, Any]], float, str, bool], None, None]:
        start = time.perf_counter()
        question = (question or "").strip()
        intake_result = self.intake.classify(question)

        if intake_result.intent == "greeting":
            answer = "Hello! How can I assist you today?" if intake_result.language == "en" else "مرحباً! كيف يمكنني مساعدتك اليوم؟"
            total = time.perf_counter() - start
            yield answer, f"{total:.2f}s", [], 1.0, "greeting", True
            return

        if not intake_result.should_retrieve:
            answer = self.validator.fallback(intake_result.language)
            total = time.perf_counter() - start
            yield answer, f"{total:.2f}s", [], 0.0, "fallback", False
            return

        # FAQ route first because it is faster and more deterministic.
        t_faq = time.perf_counter()
        faq_answer = None
        faq_hit = None
        try:
            faq_answer, faq_hit = self.retrieval.best_faq_answer(question)
        except Exception as exc:
            print(f"[FAQ retrieval error] {type(exc).__name__}: {exc}")
        faq_s = time.perf_counter() - t_faq

        if faq_answer:
            confidence = confidence_from_hits([faq_hit.get("similarity") if faq_hit else None])
            total = time.perf_counter() - start
            final = faq_answer.strip() + "\n\nSource: FAQ"
            latency = f"{total:.2f}s (FAQ {faq_s:.2f}s)"
            yield final, latency, [faq_hit] if faq_hit else [], confidence, "faq", True
            return

        # PDF RAG fallback.
        t_retrieval = time.perf_counter()
        try:
            rag_hits = self.retrieval.retrieve_pdf(question)
        except Exception as exc:
            print(f"[RAG retrieval error] {type(exc).__name__}: {exc}")
            rag_hits = []
        retrieval_s = time.perf_counter() - t_retrieval

        confidence = confidence_from_hits([hit.get("similarity") for hit in rag_hits])
        grounded = self.validator.has_reliable_context(rag_hits)

        if not grounded:
            final = self.validator.fallback(intake_result.language)
            total = time.perf_counter() - start
            latency = f"{total:.2f}s (FAQ {faq_s:.2f}s, RAG {retrieval_s:.2f}s)"
            yield final, latency, rag_hits, confidence, "fallback", False
            return

        pdf_context = "\n\n".join(
            f"[page {(hit.get('page') + 1) if hit.get('page') is not None else '?'}] {hit.get('content', '')}"
            for hit in rag_hits
        )

        t_llm = time.perf_counter()
        final = ""
        try:
            for partial in self.response.stream_answer(
                question=question,
                history=history or [],
                lang=intake_result.language,
                pdf_context=pdf_context,
            ):
                final = partial
                yield final, "", rag_hits, confidence, "pdf_rag", True
        except Exception as exc:
            error_message = f"LLM error: {type(exc).__name__}: {exc}"
            yield error_message, "", rag_hits, confidence, "error", False
            return

        llm_s = time.perf_counter() - t_llm
        pages = sorted({hit.get("page") + 1 for hit in rag_hits if hit.get("page") is not None})
        if pages and "source" not in final.lower():
            final += "\n\nSources pages: " + ", ".join(str(page) for page in pages)

        total = time.perf_counter() - start
        latency = f"{total:.2f}s (FAQ {faq_s:.2f}s, RAG {retrieval_s:.2f}s, LLM {llm_s:.2f}s)"
        yield final, latency, rag_hits, confidence, "pdf_rag", True
