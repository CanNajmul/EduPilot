"""Follow-up question rewriting (E2)."""
from __future__ import annotations

from core import prompts
from core.schemas import RewrittenQuery


def rewrite(llm, history: list[str], question: str) -> str:
    """history: recent lines like 'Student: ...' / 'EduPilot: ...' (oldest first, excluding `question`).

    With no history the question is already standalone, so no LLM call is made.
    Any failure falls back to the original question.
    """
    if not history:
        return question
    user = "Conversation so far:\n" + "\n".join(history) + f"\n\nLatest message: {question}"
    try:
        result = llm.structured(RewrittenQuery, prompts.REWRITE_SYSTEM_V1, user)
        return result.query.strip() or question
    except Exception:
        return question
