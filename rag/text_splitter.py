"""Paragraph-aware chunking and heading-based topic detection (no external deps)."""
from __future__ import annotations

import re

_HEADING_PATTERNS = [
    re.compile(r"^(chapter|unit|section|lesson)\s+\d+\b.*", re.IGNORECASE),
    re.compile(r"^\d+(\.\d+){0,2}[.)]?\s+[A-Z][^.!?]{2,80}$"),
]



def detect_heading(line: str) -> str | None:
    line = line.strip()

    if not (3 <= len(line) <= 90):
        return None

    normalized = re.sub(r"\s+", " ", line)

    # Chapter, Unit, Section and Lesson headings
    for pat in _HEADING_PATTERNS:
        if pat.match(line):
            first, *rest = normalized.split(maxsplit=1)
            return first.capitalize() + (" " + rest[0] if rest else "")

    # All-uppercase headings
    if line.isupper() and any(ch.isalpha() for ch in line) and len(line.split()) <= 10:
        return line.title()

    # Title Case headings: Machine Learning, Artificial Intelligence
    words = normalized.split()

    if (
        2 <= len(words) <= 8
        and all(
            word.isalpha()
            and word[0].isupper()
            and word[1:].islower()
            for word in words
        )
    ):
        return normalized

    return None



def page_topic(text: str, current: str) -> str:
    """Topic for a page = last heading found on it, else the topic carried over."""
    topic = current
    for line in text.splitlines():
        h = detect_heading(line)
        if h:
            topic = h
    return topic


def split_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Greedy paragraph packing with character overlap between chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > size:  # hard-split very long paragraphs
            head, para = para[:size], para[size - overlap:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
    if current:
        chunks.append(current)
    return chunks
