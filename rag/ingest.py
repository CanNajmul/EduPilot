"""Ingestion: file -> pages -> topic-tagged chunks -> Qdrant + SQLite."""
from __future__ import annotations

import shutil
from pathlib import Path
import hashlib
from config import settings
from rag.document_loader import load_document
from rag.text_splitter import page_topic, split_text
from safety.guards import looks_like_injection


def build_chunks(pages: list[dict], source: str, user_id: str, subject_id: int, doc_id: int) -> tuple[list[dict], int]:
    """Pure function: returns (chunk dicts with text + metadata, number flagged as injection)."""
    chunks: list[dict] = []
    flagged = 0
    topic = "General"
    for page in pages:
        topic = page_topic(page["text"], topic)
        for piece in split_text(page["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
            is_flagged = looks_like_injection(piece)
            flagged += int(is_flagged)
            chunks.append({
                "text": piece,
                "metadata": {
                    "user_id": user_id, "subject_id": subject_id, "doc_id": doc_id,
                    "source": source, "page": page["page"], "topic": topic, "flagged": is_flagged,
                },
            })
    return chunks, flagged



def ingest_file(src_path: str | Path, user_id: str, subject_id: int, kind: str, db, store=None) -> dict:
    """Copies the upload, indexes it (study material only), records it in SQLite."""

    settings.ensure_dirs()
    src = Path(src_path)

    if not src.is_file():
        raise FileNotFoundError(f"Upload file not found: {src}")

    file_hash = hashlib.sha256(src.read_bytes()).hexdigest()

    existing = db.find_document_by_hash(user_id, subject_id, file_hash)

    if existing:
        if existing["processed"] or existing["kind"] == "past_paper":
            return {
                "doc_id": existing["id"],
                "chunks": existing["n_chunks"],
                "flagged": existing["flagged_chunks"],
                "topics": [],
                "duplicate": True,
            }
        raise ValueError(
            "This file was uploaded before, but processing was not completed. "
            "Please resolve the previous upload before trying again."
        )

    dest_dir = settings.UPLOAD_DIR / user_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    if src.resolve() != dest.resolve():
        shutil.copy(src, dest)

    pages = load_document(dest)
    if not pages:
        raise ValueError(
            "No text could be extracted. Scanned PDFs need OCR, which is not supported yet."
        )

    doc_id = db.add_document(
        user_id, subject_id, src.name, str(dest),
        kind, file_hash=file_hash
    )

    if kind == "past_paper":
        return {
            "doc_id": doc_id,
            "chunks": 0,
            "flagged": 0,
            "topics": [],
        }

    chunks, flagged = build_chunks(
        pages, src.name, user_id, subject_id, doc_id
    )

    if store is None:
        from rag.vector_store import get_store
        store = get_store()

    store.add_texts(
        [c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    topics: list[str] = []
    for c in chunks:
        topic = c["metadata"]["topic"]
        if topic not in topics:
            topics.append(topic)

    db.add_topics(user_id, subject_id, topics)

    db._exec(
        "UPDATE documents SET n_chunks=?, flagged_chunks=?, processed=1 WHERE id=?",
        (len(chunks), flagged, doc_id),
    )

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "flagged": flagged,
        "topics": topics,
    }