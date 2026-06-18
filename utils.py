from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from config import Settings


ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATENCY_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)s")


SMALLTALK = {
    "hi",
    "hello",
    "hey",
    "السلام عليكم",
    "مرحبا",
    "هلا",
    "هاي",
    "شكرا",
    "thanks",
    "thank you",
}


def detect_lang(text: str) -> str:
    return "ar" if ARABIC_RE.search(text or "") else "en"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_smalltalk(text: str) -> bool:
    return normalize_text(text) in SMALLTALK


def now_string(settings: Settings) -> str:
    dt = datetime.now(settings.timezone)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + f" ({settings.timezone_name})"


def format_history(history: list[tuple[str, str]] | None, max_turns: int) -> str:
    if not history:
        return "(none)"

    lines: list[str] = []
    for user_msg, assistant_msg in history[-max_turns:]:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")

    return "\n".join(lines)


def parse_latency_to_seconds(latency_str: str) -> float | None:
    try:
        match = LATENCY_RE.match(latency_str or "")
        return round(float(match.group(1)), 3) if match else None
    except Exception:
        return None


def confidence_from_hits(similarities: Iterable[float | None]) -> float:
    values = [float(v) for v in similarities if v is not None]
    if not values:
        return 0.0
    best = max(values)
    # Clamp because different vector functions may return values slightly outside expected range.
    return round(max(0.0, min(1.0, best)), 3)
