"""Central configuration + graceful key resolution.

Keys resolve in this order:
  1. explicit .env / environment variable
  2. Rafael's local convention  ~/.config/<provider>/key  (raw single-line key)
This lets the demo run with zero .env edits if the OpenRouter key already exists.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "qdrant_data"
CHUNKS_PATH = BASE_DIR / "qdrant_data" / "chunks.json"
TRACE_DIR = BASE_DIR / ".traces"
CONFIG_DIR = Path.home() / ".config"


def _read_local_key(provider: str) -> str:
    p = CONFIG_DIR / provider / "key"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    # ---- LLM ----
    ronin_llm_provider: str = "openrouter"          # openrouter | anthropic | bedrock
    ronin_llm_model: str = "anthropic/claude-sonnet-4.5"
    openrouter_api_key: str = ""
    anthropic_api_key: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # ---- Embeddings ----
    ronin_embed_provider: str = "local"             # local | voyage
    voyage_api_key: str = ""
    ronin_voyage_embed_model: str = "voyage-3.5"
    ronin_voyage_rerank_model: str = "rerank-2.5"

    # ---- Observability ----
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- Slack ----
    slack_webhook_url: str = ""

    # ---------- resolvers ----------
    def llm_key(self) -> str:
        if self.ronin_llm_provider == "openrouter":
            return self.openrouter_api_key or _read_local_key("openrouter")
        if self.ronin_llm_provider == "anthropic":
            return self.anthropic_api_key or _read_local_key("anthropic")
        return ""  # bedrock uses aws creds

    def voyage_key(self) -> str:
        return self.voyage_api_key or _read_local_key("voyage")

    @property
    def embed_provider(self) -> str:
        # auto-downgrade to local if voyage selected but no key present
        if self.ronin_embed_provider == "voyage" and not self.voyage_key():
            return "local"
        return self.ronin_embed_provider

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
