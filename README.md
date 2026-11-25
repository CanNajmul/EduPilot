# EduPilot: AI-Powered Personalized Study Assistant

Upload your own study material and past exam papers. EduPilot answers with citations, makes quizzes,
finds your weak topics, analyses past-paper patterns, builds a study plan and remembers your progress.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then add GOOGLE_API_KEY and GROQ_API_KEY
streamlit run app.py
```

Free keys: Google AI Studio (Gemini), Groq, and LangSmith (tracing). Use `BAAI/bge-m3` as
`EMBEDDING_MODEL` if your material is in Bangla or mixed Bangla/English.

First run downloads the embedding model (a few hundred MB). Tests do not need it:

```bash
pytest          # 29 tests need no API keys; test_workflow.py needs langgraph installed
```

## How to use

1. Add a subject, upload a PDF/TXT/MD as **Study material** (and past papers as **Past exam paper**).
2. **Chat**: ask questions, follow up ("show me its derivation"), say "make a quiz on X",
   "what should I study next?", "make a study plan", "analyze the previous question papers",
   "share my progress report" (pauses for your approval).
3. **Quiz** tab: generate, answer, get graded. **Dashboard**: topic scores and next step.
   **Past papers**: frequency/priority report, and you can correct any topic label.

## Where each course concept lives

| Concept | Code |
|---|---|
| C1 Prompts + structured output | `core/prompts.py`, `core/schemas.py` |
| C2 Model layer, fallback, tracing | `core/llm.py` (Gemini primary, Groq `.with_fallbacks()`), LangSmith env in `config/settings.py` |
| C3 Tools | `agents/tools.py` (6 tools, all scoped by `user_id`) |
| C4 Conversation memory | `SqliteSaver` + `thread_id` in `app.py` |
| C5 Guardrails + approval | `safety/guards.py`, `safety/approval.py` (`interrupt()`), delete confirmation in `app.py` |
| C6 RAG + citations | `rag/` (PyMuPDF, topic-tagged chunks, Qdrant local mode, similarity threshold) |
| C7 LangGraph | `agents/workflow.py` |
| E1 Long-term memory | `agents/memory.py` (profile in SQLite, loaded and updated every turn) |
| E2 Follow-up rewriting | `agents/question_rewriter.py` |

Design notes: scores, frequencies, priorities and the study schedule are computed in code, not by the
LLM. The LLM does language work (answers, quiz writing, question extraction, short-answer grading).
Tools take `user_id` as an argument (the proposal listed some without it) so every access is scoped.
Topics come from headings detected at ingest, and the same topic list drives quizzes, weak-topic
tracking, past-paper classification and the plan.

## Demo script (5-7 minutes)

1. Upload a chapter PDF; show chunk/topic counts.
2. Ask a question; show citations. Ask something not in the material; show the refusal.
3. Follow-up question ("What are its types?"); show the rewritten query in the LangSmith trace.
4. Generate and take a quiz; show the Dashboard weak topic.
5. Upload 2-3 past papers; run the analysis; show priority ranking and the disclaimer.
6. Create a study plan; ask "what should I study next?".
7. "Share my progress report": graph pauses, click Approve, report exports.
8. Close the app, reopen it, ask a follow-up (memory survives). Break `GOOGLE_API_KEY` to show the Groq fallback.

## Limitations

- Scanned PDFs need OCR (not included). Formulas and tables extract poorly from some PDFs.
- Topic detection relies on headings; unstructured notes fall into a single "General" topic.
- Past-paper analysis shows historical frequency only, never a prediction.
- Level labels need at least 3 graded answers per topic.
- Designed for a few students on one machine (SQLite + local Qdrant), not a production deployment.
