"""EduPilot Streamlit app. Run: streamlit run app.py"""
from __future__ import annotations

import sqlite3
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from config import settings

settings.ensure_dirs()
settings.enable_langsmith()

from database.db import Database  # noqa: E402

st.set_page_config(page_title="EduPilot", page_icon="🎓", layout="wide")


@st.cache_resource
def get_services():
    from langchain_core.messages import HumanMessage  # noqa: F401  (fail early if missing)
    from langgraph.checkpoint.sqlite import SqliteSaver

    from agents.tools import build_tools
    from agents.workflow import build_graph
    from core.llm import get_llm
    from rag.retriever import Retriever

    db = Database(settings.DB_PATH)
    llm = get_llm()
    retriever = Retriever()
    saver = SqliteSaver(sqlite3.connect(str(settings.CHECKPOINT_PATH), check_same_thread=False))
    graph = build_graph(db, llm, retriever, checkpointer=saver)
    return db, llm, retriever, graph, build_tools(db, retriever, llm)


try:
    db, llm, retriever, graph, tools = get_services()
except Exception as exc:  # missing packages / keys: show a clear message instead of a stack trace
    st.error(f"EduPilot could not start: {exc}")
    st.info("Check that requirements are installed and GOOGLE_API_KEY / GROQ_API_KEY are set in .env.")
    st.stop()

# ---- sidebar: user, subject, uploads --------------------------------------------------
st.sidebar.title("🎓 EduPilot")
user_id = st.sidebar.text_input("Your name / ID", value=st.session_state.get("user_id", "student1")).strip()
st.session_state["user_id"] = user_id
if not user_id:
    st.stop()
db.ensure_user(user_id)

subjects = db.list_subjects(user_id)
new_subject = st.sidebar.text_input("New subject")
if st.sidebar.button("Add subject") and new_subject.strip():
    db.add_subject(user_id, new_subject.strip())
    st.rerun()
if not subjects:
    st.info("Add a subject in the sidebar to begin.")
    st.stop()

subject = st.sidebar.selectbox("Subject", subjects, format_func=lambda s: s["name"])
subject_id = subject["id"]

# ---- delete subject -----------------------------------------------------------
with st.sidebar.expander("Delete subject"):
    st.warning(
        "Deleting a subject will permanently remove its documents, "
        "topics, quizzes, study plans and related data."
    )

    confirm_name = st.text_input(
        f"Type '{subject['name']}' to confirm deletion",
        key=f"confirm_subject_{subject_id}",
    )

    if st.button(
        "Delete this subject",
        key=f"delete_subject_{subject_id}",
        disabled=confirm_name.strip() != subject["name"],
    ):
        try:
            from qdrant_client import models
            from rag.vector_store import _client

            # Remove this subject's vectors from Qdrant
            client = _client()

            if client.collection_exists(settings.COLLECTION_NAME):
                client.delete(
                    collection_name=settings.COLLECTION_NAME,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.user_id",
                                    match=models.MatchValue(value=user_id),
                                ),
                                models.FieldCondition(
                                    key="metadata.subject_id",
                                    match=models.MatchValue(value=subject_id),
                                ),
                            ]
                        )
                    ),
                    wait=True,
                )

            # Delete database records
            deleted_docs = db.delete_subject(user_id, subject_id)

            # Remove files only if no other subject uses the same path
            upload_root = (settings.UPLOAD_DIR / user_id).resolve()

            for doc in deleted_docs:
                if not doc.get("path"):
                    continue

                file_path = Path(doc["path"]).resolve()

                remaining = db._all(
                    "SELECT 1 FROM documents WHERE path=? LIMIT 1",
                    (doc["path"],),
                )

                if (
                    not remaining
                    and file_path.is_relative_to(upload_root)
                    and file_path.is_file()
                ):
                    file_path.unlink()

            st.sidebar.success("Subject deleted successfully!")
            st.rerun()

        except Exception as exc:
            st.sidebar.error(f"Could not delete subject: {exc}")
st.sidebar.subheader("Upload")
kind_label = st.sidebar.radio("Type", ["Study material", "Past exam paper"])
kind = "material" if kind_label == "Study material" else "past_paper"
upload = st.sidebar.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"])

if upload and st.sidebar.button("Process file"):
    from rag.ingest import ingest_file

    with st.spinner("Processing..."):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / upload.name
            p.write_bytes(upload.getvalue())

            try:
                info = ingest_file(p, user_id, subject_id, kind, db)

                if info.get("duplicate"):
                    st.sidebar.warning(
                        f"{upload.name} has already been uploaded for this subject. "
                        "Duplicate upload skipped."
                    )
                else:
                    msg = f"Added {upload.name}."

                    if kind == "material":
                        msg += f" {info['chunks']} chunks, {len(info['topics'])} topics."
                        if info["flagged"]:
                            msg += (
                                f" {info['flagged']} chunks were excluded "
                                "(possible prompt injection)."
                            )

                    st.sidebar.success(msg)

            except Exception as exc:
                st.sidebar.error(str(exc))

# ---- chat helpers (graph with human-in-the-loop) -------------------------------------------

# ---- uploaded files management ------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Uploaded Files")

documents = db.list_documents(user_id, subject_id)

if documents:
    selected_doc = st.sidebar.selectbox(
        "Select a file to delete",
        documents,
        format_func=lambda d: f"{d['filename']} (ID: {d['id']})",
        key="delete_document_select",
    )

    confirm_delete = st.sidebar.checkbox(
        "I confirm that I want to delete this file.",
        key="confirm_delete_document",
    )

    if st.sidebar.button(
        "Delete selected file",
        disabled=not confirm_delete,
        key="delete_document_button",
    ):
        try:
            from qdrant_client import models
            from rag.vector_store import _client

            # Delete only this document's Qdrant chunks
            client = _client()

            if client.collection_exists(settings.COLLECTION_NAME):
                client.delete(
                    collection_name=settings.COLLECTION_NAME,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.user_id",
                                    match=models.MatchValue(value=user_id),
                                ),
                                models.FieldCondition(
                                    key="metadata.subject_id",
                                    match=models.MatchValue(value=subject_id),
                                ),
                                models.FieldCondition(
                                    key="metadata.doc_id",
                                    match=models.MatchValue(value=selected_doc["id"]),
                                ),
                            ]
                        )
                    ),
                )

            # Delete extracted past-paper questions
            db._exec(
                "DELETE FROM question_patterns WHERE user_id=? AND subject_id=? AND document_id=?",
                (user_id, subject_id, selected_doc["id"]),
            )

            # Delete SQLite document record
            db._exec(
                "DELETE FROM documents WHERE id=? AND user_id=? AND subject_id=?",
                (selected_doc["id"], user_id, subject_id),
            )

            # Delete uploaded file only inside this user's upload folder
            file_path = Path(selected_doc["path"]).resolve()
            upload_root = (settings.UPLOAD_DIR / user_id).resolve()

            if file_path.is_relative_to(upload_root) and file_path.is_file():
                file_path.unlink()

            st.sidebar.success("File deleted successfully.")
            st.rerun()

        except Exception as exc:
            st.sidebar.error(f"Could not delete file: {exc}")

else:
    st.sidebar.info("No uploaded files for this subject.")

def thread_config() -> dict:
    key = f"thread_{user_id}_{subject_id}"
    if key not in st.session_state:
        st.session_state[key] = f"{user_id}-{subject_id}-{uuid.uuid4().hex[:8]}"
    return {"configurable": {"thread_id": st.session_state[key]}}


def pending_interrupt(cfg: dict):
    state = graph.get_state(cfg)
    for task in getattr(state, "tasks", []) or []:
        for it in getattr(task, "interrupts", []) or []:
            return it.value
    return None


tab_chat, tab_quiz, tab_dash, tab_patterns, tab_plan = st.tabs(
    ["💬 Chat", "📝 Quiz", "📊 Dashboard", "🗂 Past papers", "📅 Study plan"])

# ---- Chat ----------------------------------------------------------------------------------
with tab_chat:
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    cfg = thread_config()
    history = graph.get_state(cfg).values.get("messages", [])
    for m in history:
        with st.chat_message("user" if m.type == "human" else "assistant"):
            st.write(m.content)

    waiting = pending_interrupt(cfg)
    if waiting:
        st.warning("EduPilot needs your approval before sharing this report.")
        edited = st.text_area("Report", waiting.get("report", ""), height=200)
        c1, c2, c3 = st.columns(3)
        decision = None
        if c1.button("Approve"):
            decision = {"decision": "approve"}
        if c2.button("Approve with my edits"):
            decision = {"decision": "edit", "edited_report": edited}
        if c3.button("Reject"):
            decision = {"decision": "reject"}
        if decision:
            graph.invoke(Command(resume=decision), cfg)
            st.rerun()
    else:
        question = st.chat_input("Ask about your materials, request a quiz, or say 'what should I study next?'")
        if question:
            with st.spinner("Thinking..."):
                try:
                    graph.invoke(
                        {
                            "messages": [HumanMessage(content=question)],
                            "question": question,
                            "user_id": user_id,
                            "subject_id": subject_id,
                        },
                        cfg,
                    )
                    st.rerun()

                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")

    export = graph.get_state(cfg).values.get("export_path")
    if export and Path(export).exists():
        st.download_button("Download exported report", Path(export).read_bytes(), file_name=Path(export).name)

# ---- Quiz ----------------------------------------------------------------------------------
with tab_quiz:
    topic_names = [t["name"] for t in db.list_topics(user_id, subject_id)]
    if not topic_names:
        st.info("Upload study material to get topics.")
    else:
        c1, c2, c3 = st.columns(3)
        topic = c1.selectbox("Topic", topic_names)
        difficulty = c2.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        count = c3.slider("Questions", 1, settings.MAX_QUIZ_QUESTIONS, 5)
        if st.button("Generate quiz"):
            try:
                with st.spinner("Generating..."):
                    res = tools["generate_quiz"].invoke({"topic": topic, "user_id": user_id, "subject_id": subject_id,
                                                         "difficulty": difficulty, "question_count": count})
                st.session_state["quiz"] = res
                st.session_state.pop("quiz_result", None)
            except Exception as exc:
                st.error(str(exc))

        quiz = st.session_state.get("quiz")
        if quiz:
            with st.form("quiz_form"):
                answers = []
                for i, q in enumerate(quiz["quiz"]["questions"]):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    if q["options"]:
                        answers.append(st.radio("Answer", q["options"], key=f"a{i}", index=None,
                                                label_visibility="collapsed") or "")
                    else:
                        answers.append(st.text_input("Answer", key=f"a{i}", label_visibility="collapsed"))
                submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state["quiz_result"] = tools["record_quiz_result"].invoke(
                    {"quiz_id": quiz["quiz_id"], "answers": answers, "user_id": user_id, "subject_id": subject_id})
            result = st.session_state.get("quiz_result")
            if result:
                st.success(f"Score: {result['score']:.0%}")
                for i, r in enumerate(result["rows"], start=1):
                    icon = "✅" if r["score"] >= 0.99 else ("🟡" if r["score"] > 0 else "❌")
                    st.markdown(f"{icon} **Q{i}** - correct answer: {r['correct_answer']}  \n{r['explanation']}")

# ---- Dashboard -----------------------------------------------------------------------------
with tab_dash:
    prog = tools["get_student_progress"].invoke({"user_id": user_id, "subject_id": subject_id})
    perf = prog["performance"]
    if perf:
        import pandas as pd

        df = pd.DataFrame(perf).set_index("topic")[["score"]]
        st.bar_chart(df)
        st.dataframe(pd.DataFrame(perf), use_container_width=True)
    else:
        st.info("No quiz results yet.")
    nxt = prog["next"]
    st.subheader("Next step")
    st.write(f"**{nxt['action'].replace('_', ' ').title()}**: {nxt['topic'] or 'mixed quiz'}. {nxt['reason']}")
    with st.expander("Your long-term profile"):
        st.json(prog["profile"])
    with st.expander("Danger zone"):
        confirm = st.checkbox("I understand this erases my quiz history, plans and profile")
        if st.button("Delete my learning history", disabled=not confirm):
            db.delete_learning_history(user_id)
            st.success("Learning history deleted.")
            st.rerun()

# ---- Past papers -----------------------------------------------------------------------------
with tab_patterns:
    if st.button("Analyse past papers"):
        with st.spinner("Extracting questions..."):
            try:
                res = tools["analyze_question_patterns"].invoke({"user_id": user_id, "subject_id": subject_id})
                from learning.question_analyzer import render_report

                st.text(render_report(res["report"], res["priorities"]))
            except Exception as exc:
                st.error(str(exc))
    patterns = db.list_patterns(user_id, subject_id)
    if patterns:
        st.subheader("Check the classification")
        topic_names = ["Unclassified"] + [t["name"] for t in db.list_topics(user_id, subject_id)]
        for p in patterns[:30]:
            c1, c2 = st.columns([4, 2])
            c1.caption(p["text"][:200])
            new = c2.selectbox("Topic", topic_names, index=topic_names.index(p["topic"]) if p["topic"] in topic_names else 0,
                               key=f"pt{p['id']}", label_visibility="collapsed")
            if new != p["topic"]:
                db.update_pattern_topic(user_id, p["id"], new)

# ---- Study plan ----------------------------------------------------------------------------
with tab_plan:
    c1, c2 = st.columns(2)
    days = c1.number_input("Days until exam", 1, 60, 7)
    minutes = c2.number_input("Minutes per day", 30, 600, 90, step=30)
    if st.button("Create plan"):
        tools["create_study_plan"].invoke({"user_id": user_id, "subject_id": subject_id,
                                           "days": int(days), "minutes_per_day": int(minutes)})
    plan = db.latest_plan(user_id, subject_id)
    if plan:
        from core.schemas import StudyPlan
        from learning.study_planner import render_plan

        st.text(render_plan(StudyPlan(**plan)))
