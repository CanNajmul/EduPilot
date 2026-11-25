"""Central configuration. Everything tunable lives here."""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("EDUPILOT_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "edupilot.db"
CHECKPOINT_PATH = DATA_DIR / "checkpoints.db"
QDRANT_PATH = DATA_DIR / "qdrant"

# Models (C2)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_LLM_RETRIES = 0

# RAG (C6)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "edupilot_chunks"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# Guardrails (C5)
MAX_QUESTION_CHARS = 1500
MAX_QUIZ_QUESTIONS = 15
MAX_HISTORY_MESSAGES = 6

# Learning logic
MIN_ANSWERS_FOR_LEVEL = 3   # fewer graded answers than this => "insufficient data"
RECENT_ANSWERS_WINDOW = 20  # only the latest N answers per topic count
WEAK_BELOW = 0.5
STRONG_FROM = 0.75


def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOAD_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def enable_langsmith() -> None:
    """LangSmith reads these env vars; call once at startup."""
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "edupilot")
