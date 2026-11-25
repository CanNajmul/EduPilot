"""Quiz grading. MCQ and true/false are graded in code; short answers use the LLM (with a local fallback)."""
from __future__ import annotations

import re
from typing import Callable, Optional

from core.schemas import Quiz, QuizQuestion


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def grade_choice(q: QuizQuestion, user_answer: str) -> float:
    return 1.0 if _norm(user_answer) == _norm(q.answer) else 0.0


def overlap_score(expected: str, given: str) -> float:
    """Offline fallback for short answers: share of expected keywords present."""
    words = {w for w in re.findall(r"[a-zA-Z0-9]+", expected.lower()) if len(w) > 3}
    if not words:
        return 1.0 if _norm(expected) == _norm(given) else 0.0
    got = set(re.findall(r"[a-zA-Z0-9]+", given.lower()))
    return round(len(words & got) / len(words), 2)



def evaluate_quiz(
    quiz: Quiz,
    answers: list[str],
    short_grader: Optional[Callable[[QuizQuestion, str], float]] = None
) -> list[dict]:
    """Evaluate every quiz question and reject incomplete submissions."""

    if len(answers) != len(quiz.questions):
        raise ValueError(
            f"Expected {len(quiz.questions)} answers, "
            f"but received {len(answers)}."
        )

    rows = []

    for q, given in zip(quiz.questions, answers):
        if q.qtype in ("mcq", "true_false"):
            score = grade_choice(q, given)
        else:
            score = (
                short_grader(q, given)
                if short_grader
                else overlap_score(q.answer, given)
            )
            score = max(0.0, min(1.0, float(score)))

        rows.append({
            "topic": q.topic or quiz.topic,
            "question": q.question,
            "correct_answer": q.answer,
            "user_answer": given,
            "score": score,
            "explanation": q.explanation,
        })

    return rows



def total_score(rows: list[dict]) -> float:
    return sum(r["score"] for r in rows) / len(rows) if rows else 0.0
