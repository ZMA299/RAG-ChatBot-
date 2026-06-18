from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Intent = Literal[
    "greeting",
    "faq",
    "document_question",
    "administrative",
    "out_of_scope",
]


@dataclass
class IntakeResult:
    language: Literal["en", "ar"]
    intent: Intent
    normalized_question: str
    should_retrieve: bool


@dataclass
class RetrievalHit:
    source_type: Literal["faq", "pdf"]
    content: str
    page: int | None = None
    similarity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    answer: str
    latency: str
    sources_pages: list[int]
    confidence: float
    grounded: bool
    route: str
    raw_hits: list[dict[str, Any]] = field(default_factory=list)
