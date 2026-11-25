"""Semantic retrieval with a mandatory user/subject filter."""
from __future__ import annotations

from typing import Optional

from config import settings


class Retriever:
    def __init__(self, store=None, k: int = settings.TOP_K):
        self._store = store
        self.k = k

    @property
    def store(self):
        if self._store is None:
            from rag.vector_store import get_store

            self._store = get_store()
        return self._store

    def search(self, query: str, user_id: str, subject_id: int, k: Optional[int] = None,
               topic: Optional[str] = None) -> list[dict]:
        from rag.vector_store import user_filter

        results = self.store.similarity_search_with_score(
            query, k=k or self.k, filter=user_filter(user_id, subject_id)
        )
        chunks = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "topic": doc.metadata.get("topic", ""),
                "score": float(score),
            }
            for doc, score in results
        ]
        if topic:  # prefer chunks from the requested topic, keep the rest as backup
            chunks.sort(key=lambda c: (topic.lower() not in c["topic"].lower(), -c["score"]))
        return chunks
