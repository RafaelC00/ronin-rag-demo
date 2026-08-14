"""Embeddings adapter: VoyageAI (JD lib) with an offline fastembed fallback.

Default is `local` (fastembed / BGE) so the demo runs with zero external keys.
Add a free VoyageAI key and set RONIN_EMBED_PROVIDER=voyage to demo the JD stack.
"""
from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings

_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, ONNX, no torch
_LOCAL_DIM = 384
_VOYAGE_DIM = 1024


class Embedder:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.provider = self.s.embed_provider
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "voyage":
            r = self._vo.embed(texts, model=self.model, input_type="document")
            return r.embeddings
        return [list(v) for v in self._fe.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        if self.provider == "voyage":
            r = self._vo.embed([text], model=self.model, input_type="query")
            return r.embeddings[0]
        return list(next(iter(self._fe.embed([text]))))

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model} ({self.dim}d)"


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
