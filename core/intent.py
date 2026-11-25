"""Keyword intent fallback, used when the LLM classifier fails (and in tests)."""
from __future__ import annotations

import re

_RULES: list[tuple[str, str]] = [
    ("share", r"\b(share|export|send)\b.*\b(report|progress)\b|\b(report)\b.*\b(share|export)\b"),
    ("patterns", r"previous|past (year|paper)|old paper|question pattern|exam pattern|frequen"),
    ("quiz", r"\bquiz\b|\bmcqs?\b|practice questions|test me"),
    ("next_topic", r"what (should|do) i (study|learn|read) next|next topic|study next"),
    ("plan", r"study plan|schedule|timetable|\bplan\b"),
    ("performance", r"weak|strong topic|my performance|my progress|analy[sz]e my|how am i doing"),
]


def heuristic_intent(text: str) -> str:
    low = text.lower()
    for intent, pattern in _RULES:
        if re.search(pattern, low):
            return intent
    return "study"
