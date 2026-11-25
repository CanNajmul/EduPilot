"""Guardrails (C5). Pure functions so they are easy to test."""
from __future__ import annotations

import re

from config import settings

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
    r"disregard\s+.{0,40}(instructions|rules|guidelines)",
    r"forget\s+(everything|all)\s+(above|before|you were told)",
    r"you\s+are\s+now\s+(a|an|in)\s",
    r"reveal\s+.{0,30}(system prompt|hidden instructions|api key)",
    r"<\s*/?\s*(system|assistant)\s*>",
    r"new\s+instructions\s*:",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

NO_EVIDENCE_MESSAGE = (
    "I could not find this in your uploaded materials, so I won't guess. "
    "Try rephrasing, or upload the relevant chapter or notes."
)
FREQUENCY_DISCLAIMER = (
    "Note: historical frequency shows what appeared in the papers you uploaded. "
    "It is a study-priority guide, not a prediction of future exams."
)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


def check_user_input(text: str) -> tuple[bool, str]:
    """Returns (ok, message_if_blocked)."""
    if not text or not text.strip():
        return False, "Please type a question."
    if len(text) > settings.MAX_QUESTION_CHARS:
        return False, f"Please keep your message under {settings.MAX_QUESTION_CHARS} characters."
    return True, ""


def has_enough_evidence(chunks: list[dict], threshold: float | None = None) -> bool:
    threshold = settings.SIMILARITY_THRESHOLD if threshold is None else threshold
    return any(c.get("score", 0.0) >= threshold for c in chunks)



def wrap_untrusted(chunks: list[dict]) -> str:
    """Format retrieved text as clearly delimited, untrusted context."""
    parts = []

    for c in chunks:
        label = f"{c.get('source', 'unknown')}, p.{c.get('page', '?')}"
        parts.append(f"[{label}]\n{c.get('text', '')}")

    return "\n\n".join(parts)



def clamp_question_count(n: int | None, default: int = 5) -> int:
    if n is None:
        return default
    return max(1, min(int(n), settings.MAX_QUIZ_QUESTIONS))


def can_access_subject(db, user_id: str, subject_id: int) -> bool:
    return db.subject_belongs_to(user_id, subject_id)
