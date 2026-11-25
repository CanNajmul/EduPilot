"""End-to-end graph test with a fake LLM and fake retriever.

Skipped automatically when langgraph / langchain-core are not installed.
"""
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.workflow import build_graph  # noqa: E402
from core.schemas import IntentClassification, Quiz, QuizQuestion, RewrittenQuery  # noqa: E402
from database.db import Database  # noqa: E402


class FakeLLM:
    def __init__(self):
        self.rewrites = []

    def text(self, system, user):
        return "Gradient descent updates parameters against the gradient. [notes.pdf, p.1]"

    def structured(self, schema, system, user):
        if schema is RewrittenQuery:
            self.rewrites.append(user)
            return RewrittenQuery(query="What are the types of gradient descent?")
        if schema is IntentClassification:
            low = user.lower()
            if "quiz" in low:
                return IntentClassification(intent="quiz", topic="Optimisation", count=2)
            if "share" in low or "report" in low:
                return IntentClassification(intent="share")
            return IntentClassification(intent="study")
        if schema is Quiz:
            return Quiz(topic="Optimisation", questions=[
                QuizQuestion(question="q1", qtype="mcq", options=["a", "b"], answer="a"),
                QuizQuestion(question="q2", qtype="true_false", answer="True"),
            ])
        raise AssertionError(schema)


class FakeRetriever:
    def __init__(self, score=0.9):
        self.score = score

    def search(self, query, user_id, subject_id, k=5, topic=None):
        return [{"text": "Gradient descent text", "source": "notes.pdf", "page": 1, "topic": "Optimisation", "score": self.score}]


def _setup(tmp_path, score=0.9):
    db = Database(tmp_path / "t.db")
    sid = db.add_subject("u", "ML")
    db.add_topics("u", sid, ["Optimisation"])
    llm = FakeLLM()
    graph = build_graph(db, llm, FakeRetriever(score), checkpointer=MemorySaver())
    return db, sid, llm, graph


def _ask(graph, sid, text, thread="t1"):
    return graph.invoke({"messages": [HumanMessage(content=text)], "question": text, "user_id": "u", "subject_id": sid},
                        {"configurable": {"thread_id": thread}})


def test_study_question_is_grounded_and_cited(tmp_path):
    db, sid, llm, graph = _setup(tmp_path)
    out = _ask(graph, sid, "Explain gradient descent")
    assert "Sources: notes.pdf p.1" in out["messages"][-1].content


def test_low_evidence_refuses(tmp_path):
    db, sid, llm, graph = _setup(tmp_path, score=0.05)
    out = _ask(graph, sid, "Explain gradient descent")
    assert "could not find" in out["messages"][-1].content.lower()


def test_followup_is_rewritten_using_history(tmp_path):
    db, sid, llm, graph = _setup(tmp_path)
    _ask(graph, sid, "Explain gradient descent")
    out = _ask(graph, sid, "What are its types?")
    assert llm.rewrites, "second turn should call the rewriter"
    assert out["rewritten"] == "What are the types of gradient descent?"


def test_quiz_branch_saves_quiz(tmp_path):
    db, sid, llm, graph = _setup(tmp_path)
    out = _ask(graph, sid, "Make me a quiz")
    assert "Quiz #" in out["messages"][-1].content


def test_other_students_subject_is_blocked(tmp_path):
    db, sid, llm, graph = _setup(tmp_path)
    out = graph.invoke({"messages": [HumanMessage(content="hi")], "question": "hi", "user_id": "mallory", "subject_id": sid},
                       {"configurable": {"thread_id": "x"}})
    assert "own subjects" in out["messages"][-1].content


def test_share_report_pauses_then_resumes(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "EXPORT_DIR", tmp_path / "exports")
    db, sid, llm, graph = _setup(tmp_path)
    cfg = {"configurable": {"thread_id": "share"}}
    graph.invoke({"messages": [HumanMessage(content="share my report")], "question": "share my report",
                  "user_id": "u", "subject_id": sid}, cfg)
    state = graph.get_state(cfg)
    assert state.next == ("human_approval",)  # paused, nothing exported yet
    assert not (tmp_path / "exports").exists()
    out = graph.invoke(Command(resume={"decision": "approve"}), cfg)
    assert "exported" in out["messages"][-1].content.lower()
    assert any((tmp_path / "exports").iterdir())


def test_share_report_rejected(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "EXPORT_DIR", tmp_path / "exports")
    db, sid, llm, graph = _setup(tmp_path)
    cfg = {"configurable": {"thread_id": "share2"}}
    graph.invoke({"messages": [HumanMessage(content="share my report")], "question": "share my report",
                  "user_id": "u", "subject_id": sid}, cfg)
    out = graph.invoke(Command(resume={"decision": "reject"}), cfg)
    assert "did not export" in out["messages"][-1].content
