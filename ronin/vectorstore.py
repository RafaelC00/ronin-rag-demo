"""Qdrant vector store in embedded mode (on-disk, no Docker).

In production this is the exact same client pointed at a managed cluster
(Qdrant Cloud or MongoDB Atlas Vector Search) — the interface doesn't change.
"""
from __future__ import annotations

import atexit
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .config import INDEX_DIR

COLLECTION = "ronin_kb"


@lru_cache
def get_client() -> QdrantClient:
    # Embedded Qdrant locks its folder to a single client instance per process,
    # so we cache one singleton and reuse it everywhere.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(INDEX_DIR / "qdrant"))
    # close cleanly at exit so we don't hit __del__ during interpreter shutdown
    atexit.register(lambda: _safe_close(client))
    return client


def _safe_close(client: QdrantClient) -> None:
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass


def ensure_collection(client: QdrantClient, dim: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert(client: QdrantClient, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
    points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)]
    client.upsert(collection_name=COLLECTION, points=points)


def search(client: QdrantClient, query_vector: list[float], top_k: int = 8, source: str | None = None):
    flt = None
    if source:
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
    # qdrant-client >=1.14 uses query_points (the old .search is removed)
    return client.query_points(
        collection_name=COLLECTION, query=query_vector, limit=top_k, query_filter=flt, with_payload=True
    ).points
