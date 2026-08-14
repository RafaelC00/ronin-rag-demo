"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) fused with Reciprocal Rank
Fusion, then cross-encoder / Voyage reranking. Returns cited passages.

Why hybrid+rerank (2026 standard): dense catches semantics, BM25 catches exact
tokens (SKUs, "MAP", brand names); RRF fuses them; a reranker sharpens the top-k
so the generator only sees the most relevant, grounded context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .config import CHUNKS_PATH, get_settings
from .embeddings import get_embedder
from . import vectorstore as vs


@dataclass
class Passage:
    id: int
    text: str
    source: str
    title: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[{self.source}:{self.title}]"


@lru_cache
def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise RuntimeError("Index not built. Run:  python -m ronin.ingest")
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


@lru_cache
def _bm25() -> tuple[BM25Okapi, list[dict]]:
    chunks = _load_chunks()
    corpus = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(corpus), chunks


def _rrf(dense_ids: list[int], sparse_ids: list[int], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _rerank(query: str, passages: list[Passage], top_k: int) -> list[Passage]:
    s = get_settings()
    # Preferred: VoyageAI reranker (JD lib)
    if s.embed_provider == "voyage" and s.voyage_key():
        try:
            import voyageai
            vo = voyageai.Client(api_key=s.voyage_key())
            r = vo.rerank(query, [p.text for p in passages], model=s.ronin_voyage_rerank_model, top_k=top_k)
            out = []
            for item in r.results:
                p = passages[item.index]
                p.score = float(item.relevance_score)
                out.append(p)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] Voyage rerank unavailable ({e}); using local cross-encoder")
    # Fallback: local cross-encoder via fastembed
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        ce = _get_cross_encoder()
        scores = list(ce.rerank(query, [p.text for p in passages]))
        for p, sc in zip(passages, scores):
            p.score = float(sc)
        return sorted(passages, key=lambda x: x.score, reverse=True)[:top_k]
    except Exception:  # noqa: BLE001
        # Last resort: keep RRF order
        return passages[:top_k]


@lru_cache
def _get_cross_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


def retrieve(query: str, top_k: int = 4, candidates: int = 12, source: str | None = None) -> list[Passage]:
    chunks = _load_chunks()
    by_id = {c["id"]: c for c in chunks}
    embedder = get_embedder()
    client = vs.get_client()

    # dense
    qv = embedder.embed_query(query)
    dense_hits = vs.search(client, qv, top_k=candidates, source=source)
    dense_ids = [int(h.id) for h in dense_hits]

    # sparse
    bm25, _ = _bm25()
    tokenized = query.lower().split()
    bm_scores = bm25.get_scores(tokenized)
    sparse_ids = [i for i, _ in sorted(enumerate(bm_scores), key=lambda x: x[1], reverse=True)[:candidates]]
    if source:
        sparse_ids = [i for i in sparse_ids if by_id[i]["source"] == source]

    # fuse
    fused = _rrf(dense_ids, sparse_ids)
    ranked_ids = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)][:candidates]

    passages = [
        Passage(id=cid, text=by_id[cid]["text"], source=by_id[cid]["source"],
                title=by_id[cid].get("title", by_id[cid]["source"]),
                meta={k: by_id[cid].get(k) for k in ("sku", "type")})
        for cid in ranked_ids
    ]
    return _rerank(query, passages, top_k=top_k)


def format_context(passages: list[Passage]) -> str:
    return "\n\n".join(f"{p.citation}\n{p.text}" for p in passages)
