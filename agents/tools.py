"""LangChain tools (C3). Every tool takes user_id so data access is always scoped to one student."""
from __future__ import annotations

from typing import Optional

from core.schemas import Quiz
from learning import performance_analyzer as perf_mod
from learning import question_analyzer as qa
from learning import quiz_evaluator as qe
from learning import quiz_generator as qg
from learning import study_planner as sp
from agents.memory import load_profile
from safety.guards import can_access_subject


def build_tools(db, retriever, llm) -> dict:
    from langchain_core.tools import tool

    def _check(user_id: str, subject_id: int) -> None:
        if not can_access_subject(db, user_id, subject_id):
            raise PermissionError("That subject does not belong to this student.")

    @tool
    def search_materials(query: str, user_id: str, subject_id: int, k: int = 5,
                         topic: Optional[str] = None) -> list[dict]:
        """Search the student's uploaded study materials for passages relevant to the query."""
        _check(user_id, subject_id)
        return retriever.search(query, user_id=user_id, subject_id=subject_id, k=k, topic=topic)

    @tool
    def generate_quiz(topic: str, user_id: str, subject_id: int, difficulty: str = "medium",
                      question_count: int = 5) -> dict:
        """Generate and store a topic-based quiz from the student's materials. Returns quiz_id and quiz."""
        _check(user_id, subject_id)
        chunks = retriever.search(topic, user_id=user_id, subject_id=subject_id, k=8, topic=topic)
        quiz = qg.generate_quiz(llm, chunks, topic, difficulty, question_count)
        quiz_id = db.save_quiz(user_id, subject_id, topic, quiz.model_dump())
        return {"quiz_id": quiz_id, "quiz": quiz.model_dump()}

    @tool
    def record_quiz_result(quiz_id: int, answers: list[str], user_id: str, subject_id: int) -> dict:
        """Grade submitted answers for a stored quiz and record the results."""
        _check(user_id, subject_id)
        payload = db.get_quiz(user_id, quiz_id)
        if payload is None:
            raise ValueError("Quiz not found for this student.")
        quiz = Quiz.model_validate(payload)

        def short_grader(q, given):
            from core import prompts
            from core.schemas import ShortAnswerGrade
            try:
                g = llm.structured(ShortAnswerGrade, prompts.GRADE_SHORT_SYSTEM_V1,
                                   f"Question: {q.question}\nModel answer: {q.answer}\nStudent answer: {given}")
                return g.score
            except Exception:
                return qe.overlap_score(q.answer, given)

        rows = qe.evaluate_quiz(quiz, answers, short_grader)
        db.save_results(user_id, subject_id, quiz_id, rows)
        return {"score": qe.total_score(rows), "rows": rows}

    @tool
    def analyze_question_patterns(user_id: str, subject_id: int, extract: bool = True) -> dict:
        """Extract questions from uploaded past papers (if needed) and return topic frequency and priority."""
        _check(user_id, subject_id)
        if extract:
            qa.process_pending_papers(db, llm, user_id, subject_id)
        report = qa.frequency_report(db.list_patterns(user_id, subject_id))
        scores = {p.topic: p.score for p in perf_mod.analyze(db.topic_scores(user_id, subject_id))
                  if p.level != "insufficient"}
        return {"report": report, "priorities": qa.priority_scores(report, scores)}

    @tool
    def create_study_plan(user_id: str, subject_id: int, days: int = 7, minutes_per_day: int = 90) -> dict:
        """Create and store a personalised study plan from weak topics and past-paper priorities."""
        _check(user_id, subject_id)
        topics = [t["name"] for t in db.list_topics(user_id, subject_id)]
        perf = perf_mod.analyze(db.topic_scores(user_id, subject_id))
        report = qa.frequency_report(db.list_patterns(user_id, subject_id))
        scores = {p.topic: p.score for p in perf if p.level != "insufficient"}
        priorities = qa.priority_scores(report, scores) if report["total"] else None
        plan = sp.build_plan(sp.topic_weights(topics, perf, priorities), days, minutes_per_day)
        db.save_plan(user_id, subject_id, plan.model_dump())
        return plan.model_dump()

    @tool
    def get_student_progress(user_id: str, subject_id: int) -> dict:
        """Return the student's topic performance, profile and next recommended topic."""
        _check(user_id, subject_id)
        perf = perf_mod.analyze(db.topic_scores(user_id, subject_id))
        topics = db.list_topics(user_id, subject_id)
        return {
            "performance": [p.model_dump() for p in perf],
            "profile": load_profile(db, user_id),
            "next": sp.next_topic(topics, perf),
        }

    return {t.name: t for t in (search_materials, generate_quiz, record_quiz_result,
                                analyze_question_patterns, create_study_plan, get_student_progress)}
