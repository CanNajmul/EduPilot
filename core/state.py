"""LangGraph state shared by all nodes (C7)."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class EduState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # conversation (checkpointed per thread_id, C4)
    user_id: str
    subject_id: int
    question: str          # raw latest student message
    rewritten: str         # standalone query (E2)
    intent: str
    params: dict[str, Any]  # topic, count from the classifier
    context: list[dict]    # retrieved chunks
    answer: str
    blocked: bool
    profile: dict[str, Any]  # long-term memory (E1)
    report: str
    export_path: str
    subject_id: int