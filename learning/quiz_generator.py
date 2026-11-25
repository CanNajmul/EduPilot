"""LLM quiz generation grounded in retrieved material, validated in code."""
from __future__ import annotations

from core import prompts
from core.schemas import Quiz, QuizQuestion
from safety.guards import clamp_question_count, wrap_untrusted


def validate_quiz(quiz: Quiz, topic: str, max_questions: int) -> Quiz:
    """Drops malformed questions and trims to the allowed size."""
    good: list[QuizQuestion] = []
    for q in quiz.questions:
        if not q.question.strip() or not q.answer.strip():
            continue
        if q.qtype == "mcq":
            if len(q.options) < 2 or q.answer not in q.options:
                continue
        elif q.qtype == "true_false":
            q.options = ["True", "False"]
            if q.answer not in q.options:
                continue
        q.topic = q.topic or topic
        good.append(q)
    return Quiz(topic=topic, questions=good[:max_questions])


def generate_quiz(llm, chunks: list[dict], topic: str, difficulty: str = "medium",
                  count: int | None = 5) -> Quiz:
    n = clamp_question_count(count)
    system = prompts.QUIZ_SYSTEM_V1.format(count=n, difficulty=difficulty, topic=topic)
    quiz = llm.structured(Quiz, system, wrap_untrusted(chunks))
    result = validate_quiz(quiz, topic, n)
    if not result.questions:
        raise ValueError("The model returned no usable questions. Try again or choose another topic.")
    return result
