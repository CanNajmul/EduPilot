"""Previous Question Pattern Analyzer.

Extraction uses the LLM (with a regex fallback). Everything numeric
(frequencies, shares, priority) is computed in code.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Optional

from core import prompts
from core.schemas import ExtractedQuestions, QuestionPattern
from rag.document_loader import load_document, pages_to_text
from safety.guards import FREQUENCY_DISCLAIMER, wrap_untrusted

UNCLASSIFIED = "Unclassified"


# -- classification helpers (pure) ------------------------------------------
def guess_qtype(text: str) -> str:
    low = text.lower()
    if re.search(r"\(a\).*\(b\)|\ba\)\s.*\bb\)\s", low, re.DOTALL):
        return "mcq"
    if "true or false" in low or "true/false" in low:
        return "true_false"
    if re.search(r"\b(derive|prove|show that)\b", low):
        return "derivation"
    if re.search(r"\b(calculate|compute|evaluate|find the value|solve)\b", low):
        return "numerical"
    return "conceptual"


def guess_marks(text: str) -> Optional[float]:
    m = re.search(r"[\[(]\s*(\d+(?:\.\d+)?)\s*marks?\s*[\])]", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def guess_year(text: str) -> Optional[int]:
    m = re.search(r"\b(19[89]\d|20[0-4]\d)\b", text)
    return int(m.group(1)) if m else None


def normalize_topic(raw: str, topics: list[str]) -> str:
    """Force the model's topic onto the known topic list (never invent new topics)."""
    if not raw or not topics:
        return UNCLASSIFIED
    lower = {t.lower(): t for t in topics}
    if raw.lower() in lower:
        return lower[raw.lower()]
    match = difflib.get_close_matches(raw.lower(), list(lower), n=1, cutoff=0.6)
    return lower[match[0]] if match else UNCLASSIFIED


def keyword_topic(text: str, topics: list[str]) -> str:
    """Offline fallback: pick the topic whose words appear most in the question."""
    words = set(re.findall(r"[a-z]{4,}", text.lower()))
    best, best_hits = UNCLASSIFIED, 0
    for t in topics:
        hits = len(words & set(re.findall(r"[a-z]{4,}", t.lower())))
        if hits > best_hits:
            best, best_hits = t, hits
    return best


def rule_based_extract(text: str, topics: list[str], default_year: Optional[int] = None) -> list[QuestionPattern]:
    """Regex fallback: split on numbered questions such as '1.' or 'Q2)'."""
    parts = re.split(r"(?m)^\s*(?:Q(?:uestion)?\.?\s*)?\d{1,2}[.)]\s+", text)
    out = []
    for part in parts[1:]:
        q = re.sub(r"\s+", " ", part).strip()
        if len(q) < 15:
            continue
        out.append(QuestionPattern(text=q, year=guess_year(q) or default_year, marks=guess_marks(q),
                                   qtype=guess_qtype(q), topic=keyword_topic(q, topics)))
    return out


# -- extraction with the LLM -------------------------------------------------
def extract_questions(llm, text: str, topics: list[str], default_year: Optional[int] = None) -> list[QuestionPattern]:
    system = prompts.EXTRACT_QUESTIONS_SYSTEM_V1.format(topics=", ".join(topics) or "(none)")
    found: list[QuestionPattern] = []
    segments = [text[i:i + 6000] for i in range(0, len(text), 6000)] or [""]
    for seg in segments:
        try:
            res = llm.structured(ExtractedQuestions, system, wrap_untrusted([{"text": seg, "source": "paper"}]))
            found.extend(res.questions)
        except Exception:
            found.extend(rule_based_extract(seg, topics, default_year))
    for q in found:
        q.topic = normalize_topic(q.topic, topics)
        q.year = q.year or default_year
    return found


def process_pending_papers(db, llm, user_id: str, subject_id: int) -> int:
    """Extract questions from past papers that have no saved patterns yet."""
    topics = [t["name"] for t in db.list_topics(user_id, subject_id)]
    saved = 0

    existing_patterns = db.list_patterns(user_id, subject_id)
    processed_paper_ids = {
        p["document_id"] for p in existing_patterns
        if p.get("document_id") is not None
    }

    for doc in db.list_documents(user_id, subject_id, kind="past_paper"):
        if doc["id"] in processed_paper_ids:
            continue

        text = pages_to_text(load_document(doc["path"]))
        patterns = extract_questions(
            llm, text, topics, default_year=guess_year(doc["filename"])
        )

        if patterns:
            db.save_patterns(
                user_id, subject_id, doc["id"],
                [p.model_dump() for p in patterns]
            )
            saved += len(patterns)

        db.mark_document_processed(user_id, doc["id"])

    return saved


# -- numbers (pure) -----------------------------------------------------------
def frequency_report(patterns: list[dict]) -> dict:
    total = len(patterns)
    by_topic = Counter(p["topic"] or UNCLASSIFIED for p in patterns)
    by_type = Counter(p["qtype"] or "conceptual" for p in patterns)
    by_year = Counter(p["year"] for p in patterns if p.get("year"))
    return {
        "total": total,
        "by_topic": [{"topic": t, "count": c, "share": round(c / total, 3)} for t, c in by_topic.most_common()] if total else [],
        "by_type": dict(by_type.most_common()),
        "by_year": dict(sorted(by_year.items())),
    }


def priority_scores(report: dict, perf: dict[str, float], min_known: float = 0.5) -> list[dict]:
    """priority = historical share x (1 + weakness). Unknown performance counts as `min_known` weakness."""
    rows = []
    for row in report["by_topic"]:
        weakness = (1.0 - perf[row["topic"]]) if row["topic"] in perf else min_known
        rows.append({**row, "weakness": round(weakness, 3),
                     "priority": round(row["share"] * (1.0 + weakness), 4)})
    return sorted(rows, key=lambda r: r["priority"], reverse=True)




def render_report(report: dict, priorities: list[dict]) -> str:
    if not report.get("total", 0):
        return (
            "No past-paper questions found yet. "
            "Upload past papers as 'past_paper' and try again."
        )

    lines = [
        f"Analysed {report['total']} questions from your past papers.",
        "",
        "Topics by priority:"
    ]

    for r in priorities[:10]:
        lines.append(
            f"- {r['topic']}: asked {r['count']}x ({r['share']:.0%}), "
            f"your weakness {r['weakness']:.0%}, priority {r['priority']:.2f}"
        )

    lines += [
        "",
        "Question types: " + ", ".join(
            f"{k} ({v})" for k, v in report["by_type"].items()
        )
    ]

    if report["by_year"]:
        lines.append(
            "By year: " + ", ".join(
                f"{y}: {n}" for y, n in report["by_year"].items()
            )
        )

    lines += ["", FREQUENCY_DISCLAIMER]

    return "\n".join(lines)


