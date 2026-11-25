"""Study plan and next-topic logic. Scheduling is deterministic code; no LLM needed."""
from __future__ import annotations

from core.schemas import StudyDay, StudyPlan, TopicPerformance
import math
BLOCK_MINUTES = 30
_ACTIVITIES = ["Read and make notes", "Practice quiz", "Revise weak points", "Mixed recall"]


def topic_weights(topics: list[str], perf: list[TopicPerformance], priorities: list[dict] | None = None) -> dict[str, float]:
    """Higher weight = more study time. Past-paper priority wins when available."""
    if priorities:
        w = {r["topic"]: r["priority"] for r in priorities if r["topic"] in topics}
    else:
        w = {}
    by_topic = {p.topic: p for p in perf}
    for t in topics:
        if t in w:
            continue
        p = by_topic.get(t)
        if p is None or p.level == "insufficient":
            w[t] = 0.5
        elif p.level == "strong":
            w[t] = 0.1
        else:
            w[t] = (1.0 - p.score) + 0.2
    return w



def build_plan(weights: dict[str, float], days: int, minutes_per_day: int) -> StudyPlan:
    """Build a weighted study plan without exceeding daily time limits."""

    days = max(1, int(days))
    minutes_per_day = max(0, int(minutes_per_day))

    if not weights:
        return StudyPlan(
            days=[],
            summary="No topics yet. Upload study material first."
        )

    # Only complete 30-minute blocks are allowed.
    blocks_per_day = minutes_per_day // BLOCK_MINUTES

    if blocks_per_day == 0:
        return StudyPlan(
            days=[],
            summary=(
                f"No 30-minute study block fits within "
                f"{minutes_per_day} minutes per day."
            )
        )

    # Maximum available study sessions.
    blocks_total = days * blocks_per_day

    # Sanitize weights.
    clean_weights = {}

    for topic, value in weights.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = 0.0

        clean_weights[topic] = (
            weight if math.isfinite(weight) and weight > 0 else 0.0
        )

    # Use equal weights if all priorities are zero.
    if not any(clean_weights.values()):
        clean_weights = {topic: 1.0 for topic in weights}

    # Highest-priority topics come first.
    topics = sorted(
        clean_weights,
        key=clean_weights.get,
        reverse=True
    )

    # Never schedule more topics than available time slots.
    selected = topics[:blocks_total]

    # Give each selected topic one initial session.
    alloc = {topic: 1 for topic in selected}

    # Distribute remaining sessions proportionally.
    remaining = blocks_total - len(selected)

    if remaining > 0:
        total_weight = sum(clean_weights[t] for t in selected)

        exact = {
            t: remaining * clean_weights[t] / total_weight
            for t in selected
        }

        for topic in selected:
            alloc[topic] += int(exact[topic])

        leftover = remaining - sum(
            int(exact[t]) for t in selected
        )

        # Largest-remainder allocation.
        remainder_order = sorted(
            selected,
            key=lambda t: exact[t] - int(exact[t]),
            reverse=True
        )

        for topic in remainder_order[:leftover]:
            alloc[topic] += 1

    # Create the study session queue.
    queue = []

    for topic in selected:
        for i in range(alloc[topic]):
            activity = _ACTIVITIES[i % len(_ACTIVITIES)]
            queue.append((topic, activity))

    # Assign sessions within each day's capacity.
    plan_days = []

    for idx, (topic, activity) in enumerate(queue):
        day = idx // blocks_per_day + 1

        plan_days.append(
            StudyDay(
                day=day,
                topic=topic,
                activity=activity,
                minutes=BLOCK_MINUTES
            )
        )

    summary = (
        f"{days}-day plan, up to {minutes_per_day} min/day. "
        f"Scheduled {blocks_total} blocks across "
        f"{len(selected)} topics. "
        f"Most time goes to: {', '.join(selected[:3])}."
    )

    return StudyPlan(
        days=plan_days,
        summary=summary
    )



def render_plan(plan: StudyPlan) -> str:
    if not plan.days:
        return plan.summary
    lines = [plan.summary, ""]
    by_day: dict[int, list[StudyDay]] = {}
    for d in plan.days:
        by_day.setdefault(d.day, []).append(d)
    for day in sorted(by_day):
        lines.append(f"Day {day}")
        lines += [f"  - {d.minutes} min  {d.topic}: {d.activity}" for d in by_day[day]]
    return "\n".join(lines)


def next_topic(topics: list[dict], perf: list[TopicPerformance]) -> dict:
    """Rule-based: revise the weakest topic, else the first incomplete one (document order = prerequisite order)."""
    weak = [p for p in perf if p.level == "weak"]
    if weak:
        w = min(weak, key=lambda p: p.score)
        return {"topic": w.topic, "action": "revise",
                "reason": f"Your score on this topic is {w.score:.0%}. Revise before moving on."}
    for t in topics:
        if not t["completed"]:
            return {"topic": t["name"], "action": "study", "reason": "Next topic in your material that you have not completed."}
    return {"topic": None, "action": "mixed_quiz",
            "reason": "You have completed every topic. Take a mixed quiz to keep it fresh."}
