"""Embedding model. Switch EMBEDDING_MODEL to BAAI/bge-m3 for Bangla or mixed-language material."""
from __future__ import annotations

from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity in [0, 1] for typical text
    )
