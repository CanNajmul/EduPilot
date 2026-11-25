"""Topic-level performance, computed in code (not by the LLM)."""
from __future__ import annotations

from config import settings
from core.schemas import TopicPerformance


def classify_level(score: float, answers: int) -> str:
    if answers < settings.MIN_ANSWERS_FOR_LEVEL:
        return "insufficient"
    if score < settings.WEAK_BELOW:
        return "weak"
    if score >= settings.STRONG_FROM:
        return "strong"
    return "moderate"


def analyze(topic_scores: dict[str, list[float]]) -> list[TopicPerformance]:
    """topic_scores: topic -> recent answer scores in [0, 1]. Returns worst-first."""
    out = []
    for topic, scores in topic_scores.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        out.append(TopicPerformance(topic=topic, answers=len(scores), score=round(avg, 3),
                                    level=classify_level(avg, len(scores))))
    return sorted(out, key=lambda p: p.score)



def summarize(perf: list[TopicPerformance]) -> str:
    if not perf:
        return "No quiz results yet. Take a quiz first and I will analyse your performance."

    lines = []

    for p in perf:
        if p.level == "insufficient":
            lines.append(
                f"- {p.topic}: not enough answers yet ({p.answers}) to judge"
            )
        else:
            lines.append(
                f"- {p.topic}: {p.score:.0%} over {p.answers} answers ({p.level})"
            )

    weak = [p.topic for p in perf if p.level == "weak"]
    tail = f"\n\nFocus first on: {', '.join(weak)}." if weak else ""

    return "Your performance by topic (weakest first):\n" + "\n".join(lines) + tail

