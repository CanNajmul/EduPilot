
"""Human approval via LangGraph interrupt() (C5)."""
from __future__ import annotations

from typing import Any


def ask_approval(payload: dict[str, Any]) -> dict[str, Any]:
    """Pause the graph until the app resumes it with Command(resume=...)."""

    from langgraph.types import interrupt

    decision = interrupt(payload)

    if not isinstance(decision, dict):
        decision = {"decision": str(decision)}

    return decision
