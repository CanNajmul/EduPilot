"""Pydantic schemas used for structured LLM output (C1) and internal data."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["study", "quiz", "performance", "patterns", "plan", "next_topic", "share"]
QType = Literal["mcq", "true_false", "short"]
Level = Literal["weak", "moderate", "strong", "insufficient"]


class IntentClassification(BaseModel):
    intent: Intent = Field(description="What the student wants to do")
    topic: Optional[str] = Field(default=None, description="Topic mentioned, if any")
    count: Optional[int] = Field(default=None, description="Number of questions requested, if any")


class RewrittenQuery(BaseModel):
    query: str = Field(description="Standalone version of the student's latest question")


class QuizQuestion(BaseModel):
    question: str
    qtype: QType = "mcq"
    options: list[str] = Field(default_factory=list)
    answer: str
    explanation: str = ""
    topic: str = ""


class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]


class ShortAnswerGrade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = ""


class TopicPerformance(BaseModel):
    topic: str
    answers: int
    score: float
    level: Level


class StudyDay(BaseModel):
    day: int
    topic: str
    activity: str
    minutes: int


class StudyPlan(BaseModel):
    days: list[StudyDay]
    summary: str = ""


class QuestionPattern(BaseModel):
    text: str
    year: Optional[int] = None
    marks: Optional[float] = None
    qtype: str = "conceptual"
    topic: str = "Unclassified"


class ExtractedQuestions(BaseModel):
    questions: list[QuestionPattern]
