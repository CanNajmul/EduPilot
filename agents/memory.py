"""Long-term learning profile (E1), persisted in SQLite and keyed by user_id.

Equivalent to a LangGraph Store namespace ("profile", user_id); implemented on SQLite so it
survives restarts without extra infrastructure.
"""
from __future__ import annotations

from typing import Any

from learning.performance_analyzer import analyze

DEFAULT_PROFILE: dict[str, Any] = {
    "subjects": [],
    "weak_topics": [],
    "moderate_topics": [],
    "strong_topics": [],
    "completed_topics": [],
    "quiz_scores": {},
    "goals": "",
    "explanation_style": "simple and step by step",
    "last_intent": None,
}


def load_profile(db, user_id: str) -> dict[str, Any]:
    stored = db.get_profile(user_id) or {}
    return {**DEFAULT_PROFILE, **stored}


def update_profile(db, user_id: str, intent: str | None = None) -> dict[str, Any]:
    """Recomputes weak/strong topics from quiz history and saves the profile."""
    profile = load_profile(db, user_id)
    perf = analyze(db.topic_scores(user_id))
    profile["weak_topics"] = [p.topic for p in perf if p.level == "weak"]
    profile["moderate_topics"] = [p.topic for p in perf if p.level == "moderate"]
    profile["strong_topics"] = [p.topic for p in perf if p.level == "strong"]
    profile["quiz_scores"] = {p.topic: p.score for p in perf}
    profile["subjects"] = [s["name"] for s in db.list_subjects(user_id)]
    for p in perf:
        db.set_topic_completed(user_id, p.topic, p.level == "strong")
    profile["completed_topics"] = [t["name"] for t in db.list_topics(user_id) if t["completed"]]
    if intent:
        profile["last_intent"] = intent
    db.save_profile(user_id, profile)
    return profile
