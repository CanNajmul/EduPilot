"""The LangGraph workflow (C7).

START -> load_profile -> guard_input -> rewrite_query -> classify_intent -> [branch] -> update_profile -> END

Branches:
  study       retrieve_materials -> generate_answer
  quiz        make_quiz
  performance analyze_performance
  patterns    extract_questions -> calculate_frequency
  plan        make_plan
  next_topic  recommend_next
  share       prepare_report -> human_approval (interrupt) -> export_report
"""
from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from agents.memory import load_profile, update_profile
from agents.question_rewriter import rewrite
from agents.tools import build_tools
from config import settings
from core import prompts
from core.intent import heuristic_intent
from core.schemas import IntentClassification
from core.state import EduState
from learning import performance_analyzer as perf_mod
from learning import question_analyzer as qa
from learning import study_planner as sp
from safety import guards
from safety.approval import ask_approval

ROUTES = ["study", "quiz", "performance", "patterns", "plan", "next_topic", "share"]


def build_graph(db, llm, retriever, checkpointer=None):
    tools = build_tools(db, retriever, llm)

    def _args(state: EduState, **extra) -> dict:
        return {"user_id": state["user_id"], "subject_id": state["subject_id"], **extra}

    # ---- entry nodes -------------------------------------------------------
    def load_profile_node(state: EduState):
        return {"profile": load_profile(db, state["user_id"])}

    def guard_input(state: EduState):
        ok, msg = guards.check_user_input(state.get("question", ""))
        if ok and not guards.can_access_subject(db, state["user_id"], state["subject_id"]):
            ok, msg = False, "Please select one of your own subjects."
        return {"blocked": not ok, "answer": msg if not ok else ""}

    def after_guard(state: EduState) -> str:
        return "blocked" if state.get("blocked") else "ok"

    def rewrite_query(state: EduState):
        msgs = state.get("messages", [])[:-1][-settings.MAX_HISTORY_MESSAGES:]
        history = [f"{'Student' if m.type == 'human' else 'EduPilot'}: {m.content}" for m in msgs]
        return {"rewritten": rewrite(llm, history, state["question"])}

    def classify_intent(state: EduState):
        text = state["rewritten"]
        try:
            r = llm.structured(IntentClassification, prompts.INTENT_SYSTEM_V1, text)
            return {"intent": r.intent, "params": {"topic": r.topic, "count": r.count}}
        except Exception:
            return {"intent": heuristic_intent(text), "params": {"topic": None, "count": None}}

    def route_intent(state: EduState) -> str:
        intent = state.get("intent", "study")
        return intent if intent in ROUTES else "study"

    # ---- study branch ------------------------------------------------------
    def retrieve_materials(state: EduState):
        chunks = tools["search_materials"].invoke(_args(state, query=state["rewritten"]))
        return {"context": chunks}

    def generate_answer(state: EduState):
        chunks = state.get("context", [])
        if not guards.has_enough_evidence(chunks):
            return {"answer": guards.NO_EVIDENCE_MESSAGE}
        style = state.get("profile", {}).get("explanation_style", "simple")
        used = [c for c in chunks if c["score"] >= settings.SIMILARITY_THRESHOLD]
        answer = llm.text(prompts.ANSWER_SYSTEM_V1.format(style=style),
                          f"{guards.wrap_untrusted(used)}\n\nQuestion: {state['rewritten']}")
        sources = sorted({f"{c['source']} p.{c['page']}" for c in used})
        return {"answer": f"{answer}\n\nSources: {', '.join(sources)}"}

    # ---- quiz branch -------------------------------------------------------
    def make_quiz(state: EduState):
        p = state.get("params", {})
        topic = p.get("topic")
        if not topic:
            perf = perf_mod.analyze(db.topic_scores(state["user_id"], state["subject_id"]))
            topics = db.list_topics(state["user_id"], state["subject_id"])
            topic = perf[0].topic if perf and perf[0].level == "weak" else (topics[0]["name"] if topics else "")
        if not topic:
            return {"answer": "Upload some study material first so I can build a quiz."}
        try:
            res = tools["generate_quiz"].invoke(
                _args(state, topic=topic, question_count=guards.clamp_question_count(p.get("count"))))
        except Exception as exc:
            return {"answer": f"I could not build that quiz: {exc}"}
        n = len(res["quiz"]["questions"])
        return {"answer": f"Quiz #{res['quiz_id']} on '{topic}' is ready ({n} questions). Open the Quiz tab to take it."}

    # ---- performance / next topic / plan -----------------------------------
    def analyze_performance(state: EduState):
        prog = tools["get_student_progress"].invoke(_args(state))
        from core.schemas import TopicPerformance
        perf = [TopicPerformance(**p) for p in prog["performance"]]
        return {"answer": perf_mod.summarize(perf)}

    def recommend_next(state: EduState):
        prog = tools["get_student_progress"].invoke(_args(state))
        nxt = prog["next"]
        label = nxt["topic"] or "a mixed quiz"
        return {"answer": f"Suggested next step: {nxt['action']} - {label}. {nxt['reason']}"}

    def make_plan(state: EduState):
        plan = tools["create_study_plan"].invoke(_args(state))
        from core.schemas import StudyPlan
        return {"answer": sp.render_plan(StudyPlan(**plan))}

    # ---- patterns branch -----------------------------------------------------
    def extract_questions(state: EduState):
        try:
            tools["analyze_question_patterns"].invoke(_args(state, extract=True))
        except Exception as exc:
            return {"answer": f"I could not read the past papers: {exc}"}
        return {}

    def calculate_frequency(state: EduState):
        if state.get("answer", "").startswith("I could not"):
            return {}
        res = tools["analyze_question_patterns"].invoke(_args(state, extract=False))
        return {"answer": qa.render_report(res["report"], res["priorities"])}

    # ---- share branch (human in the loop) ------------------------------------
    def prepare_report(state: EduState):
        prog = tools["get_student_progress"].invoke(_args(state))
        from core.schemas import TopicPerformance
        perf = [TopicPerformance(**p) for p in prog["performance"]]
        report = f"EduPilot progress report ({datetime.now():%Y-%m-%d})\n\n{perf_mod.summarize(perf)}"
        return {"report": report}

    def human_approval(state: EduState):
        decision = ask_approval({"action": "share_report", "report": state["report"],
                                 "question": "Approve exporting this report?"})
        choice = decision.get("decision", "reject")
        if choice == "edit" and decision.get("edited_report"):
            return {"report": decision["edited_report"], "intent": "share_approved"}
        if choice == "approve":
            return {"intent": "share_approved"}
        return {"intent": "share_rejected"}

    def after_approval(state: EduState) -> str:
        return "export" if state.get("intent") == "share_approved" else "cancel"

    def export_report(state: EduState):
        settings.ensure_dirs()
        path = settings.EXPORT_DIR / f"report_{state['user_id']}_{datetime.now():%Y%m%d_%H%M%S}.md"
        path.write_text(state["report"], encoding="utf-8")
        return {"export_path": str(path), "answer": f"Report exported to {path.name}."}

    def cancel_export(state: EduState):
        return {"answer": "Okay, I did not export or share the report."}

    # ---- exit ----------------------------------------------------------------
    def finish(state: EduState):
        profile = update_profile(db, state["user_id"], state.get("intent"))
        answer = state.get("answer") or "Sorry, I could not produce an answer."
        return {"profile": profile, "messages": [AIMessage(content=answer)]}

    g = StateGraph(EduState)
    for name, fn in [
        ("load_profile", load_profile_node), ("guard_input", guard_input), ("rewrite_query", rewrite_query),
        ("classify_intent", classify_intent), ("retrieve_materials", retrieve_materials),
        ("generate_answer", generate_answer), ("make_quiz", make_quiz),
        ("analyze_performance", analyze_performance), ("recommend_next", recommend_next),
        ("make_plan", make_plan), ("extract_questions", extract_questions),
        ("calculate_frequency", calculate_frequency), ("prepare_report", prepare_report),
        ("human_approval", human_approval), ("export_report", export_report),
        ("cancel_export", cancel_export), ("finish", finish),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "load_profile")
    g.add_edge("load_profile", "guard_input")
    g.add_conditional_edges("guard_input", after_guard, {"ok": "rewrite_query", "blocked": "finish"})
    g.add_edge("rewrite_query", "classify_intent")
    g.add_conditional_edges("classify_intent", route_intent, {
        "study": "retrieve_materials", "quiz": "make_quiz", "performance": "analyze_performance",
        "patterns": "extract_questions", "plan": "make_plan", "next_topic": "recommend_next",
        "share": "prepare_report",
    })
    g.add_edge("retrieve_materials", "generate_answer")
    g.add_edge("extract_questions", "calculate_frequency")
    g.add_edge("prepare_report", "human_approval")
    g.add_conditional_edges("human_approval", after_approval, {"export": "export_report", "cancel": "cancel_export"})
    for terminal in ("generate_answer", "make_quiz", "analyze_performance", "recommend_next", "make_plan",
                     "calculate_frequency", "export_report", "cancel_export"):
        g.add_edge(terminal, "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)
