"""LLM provider adapter.

One interface, three backends. The default is `openrouter`; setting RONIN_LLM_PROVIDER=bedrock (or `anthropic`)
switches provider with no other code change, keeping the agents provider-agnostic.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings, get_settings


class LLM:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.provider = self.s.ronin_llm_provider
        self.model = self.s.ronin_llm_model
        self._client = self._build_client()

    def _build_client(self) -> Any:
        if self.provider == "openrouter":
            from openai import OpenAI
            key = self.s.llm_key()
            if not key:
                raise RuntimeError(
                    "No OpenRouter key found. Set OPENROUTER_API_KEY or create ~/.config/openrouter/key"
                )
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                default_headers={"HTTP-Referer": "https://ronin-demo.local", "X-Title": "Ronin RAG Demo"},
            )
        if self.provider == "anthropic":
            from anthropic import Anthropic  # lazy: only needed on this path
            return Anthropic(api_key=self.s.llm_key())
        if self.provider == "bedrock":
            import boto3  # lazy: only needed on this path
            return boto3.client("bedrock-runtime", region_name=self.s.aws_region)
        raise ValueError(f"Unknown provider {self.provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def complete(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Return a plain-text completion."""
        if self.provider in ("openrouter",):
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model, system=system, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        if self.provider == "bedrock":
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
                "temperature": temperature, "system": system,
                "messages": [{"role": "user", "content": user}],
            })
            resp = self._client.invoke_model(modelId=self.model, body=body)
            return json.loads(resp["body"].read())["content"][0]["text"]
        raise ValueError(self.provider)

    def json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024) -> dict:
        """Ask for strict JSON and parse it, tolerating markdown fences."""
        raw = self.complete(
            system + "\n\nRespond with ONLY valid minified JSON. No prose, no markdown fences.",
            user, temperature=temperature, max_tokens=max_tokens,
        )
        return _parse_json(raw)


@lru_cache
def get_llm() -> LLM:
    return LLM()


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start : end + 1])
        raise
