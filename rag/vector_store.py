"""Qdrant (local mode, no server needed) through langchain-qdrant."""
from __future__ import annotations

from functools import lru_cache

from config import settings
from rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def _client():
    from qdrant_client import QdrantClient

    settings.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.QDRANT_PATH))


@lru_cache(maxsize=1)
def get_store():
    from qdrant_client.models import Distance, VectorParams
    from langchain_qdrant import QdrantVectorStore

    client = _client()
    emb = get_embeddings()
    if not client.collection_exists(settings.COLLECTION_NAME):
        dim = len(emb.embed_query("dimension probe"))
        client.create_collection(
            settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=settings.COLLECTION_NAME, embedding=emb)


def user_filter(user_id: str, subject_id: int):
    """Mandatory metadata filter: a query can only ever see its own user's subject."""
    from qdrant_client import models

    return models.Filter(
        must=[
            models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="metadata.subject_id", match=models.MatchValue(value=subject_id)),
            models.FieldCondition(key="metadata.flagged", match=models.MatchValue(value=False)),
        ]
    )
