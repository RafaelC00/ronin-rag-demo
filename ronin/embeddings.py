"""Embeddings adapter: VoyageAI with an offline fastembed fallback.

Default is `local` (fastembed / BGE) so the app runs with zero external keys.
Add a VoyageAI key and set RONIN_EMBED_PROVIDER=voyage for hosted embeddings and reranking.
"""
from __future__ import annotations

import time
from functools import lru_cache

from .config import Settings, get_settings

_QUERY_CACHE_MAX = 256
_RATE_LIMIT_WAITS = (21, 21, 21)  # Voyage free tier is 3 RPM; window resets within a minute

_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, ONNX, no torch
_LOCAL_DIM = 384
_VOYAGE_DIM = 1024


class Embedder:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.provider = self.s.embed_provider
        self._query_cache: dict[str, list[float]] = {}
        if self.provider == "voyage":
            import voyageai
            self._vo = voyageai.Client(api_key=self.s.voyage_key())
            self.model = self.s.ronin_voyage_embed_model
            self.dim = _VOYAGE_DIM
        else:
            from fastembed import TextEmbedding
            self._fe = TextEmbedding(model_name=_LOCAL_MODEL)
            self.model = _LOCAL_MODEL
            self.dim = _LOCAL_DIM

    def _vo_embed(self, texts: list[str], input_type: str):
        """Voyage call with backoff on the free tier's 3 RPM rate limit."""
        from voyageai.error import RateLimitError

        for attempt, wait in enumerate((0, *_RATE_LIMIT_WAITS)):
            if wait:
                time.sleep(wait)
            try:
                return self._vo.embed(texts, model=self.model, input_type=input_type)
            except RateLimitError:
                if attempt == len(_RATE_LIMIT_WAITS):
                    raise
        raise RuntimeError("unreachable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "voyage":
            return self._vo_embed(texts, "document").embeddings
        return [list(v) for v in self._fe.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        if self.provider == "voyage":
            vec = self._vo_embed([text], "query").embeddings[0]
        else:
            vec = list(next(iter(self._fe.embed([text]))))
        self._cache_queries({text: vec})
        return vec

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Batch query embedding — one API call, fills the query cache.

        Use before loops that would otherwise call embed_query per item
        (e.g. the eval harness): 1 request instead of N against a 3 RPM tier.
        """
        missing = [t for t in texts if t not in self._query_cache]
        if missing:
            if self.provider == "voyage":
                vecs = self._vo_embed(missing, "query").embeddings
            else:
                vecs = [list(v) for v in self._fe.embed(missing)]
            self._cache_queries(dict(zip(missing, vecs)))
        return [self._query_cache[t] for t in texts]

    def _cache_queries(self, items: dict[str, list[float]]) -> None:
        for text, vec in items.items():
            if len(self._query_cache) >= _QUERY_CACHE_MAX:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[text] = vec

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model} ({self.dim}d)"


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
