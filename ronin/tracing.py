"""Observability: Langfuse when keys exist, else a local JSONL trace the UI reads.

Every agent step and LLM call is recorded (name, input, output, latency) so the
demo can *show* observability even fully offline — same discipline as Langfuse.
Supports the Langfuse v4 (OTEL) API with a v2 fallback.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

from .config import TRACE_DIR, get_settings

_settings = get_settings()
_langfuse = None
if _settings.langfuse_enabled:
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=_settings.langfuse_public_key,
            secret_key=_settings.langfuse_secret_key,
            host=_settings.langfuse_host,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[tracing] Langfuse init failed ({e}); using local traces")

TRACE_DIR.mkdir(parents=True, exist_ok=True)
_TRACE_FILE = TRACE_DIR / "trace.jsonl"


def _lf_open(name: str, input_data):
    """Return an entered Langfuse observation context manager (v4) or None.

    Uses start_as_current_observation so nested span() calls attach as children
    of the enclosing observation -> one clean trace per query.
    """
    if not _langfuse:
        return None, None
    fn = getattr(_langfuse, "start_as_current_observation", None)
    if not fn:
        return None, None
    for kwargs in ({"name": name, "as_type": "span", "input": input_data},
                   {"name": name, "input": input_data},
                   {"name": name}):
        try:
            cm = fn(**kwargs)
            obs = cm.__enter__()
            return cm, obs
        except TypeError:
            continue
        except Exception:  # noqa: BLE001
            return None, None
    return None, None


def _write_local(record: dict) -> None:
    with _TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def span(name: str, input_data=None, kind: str = "span"):
    """Trace a unit of work. Yields a dict you can attach `output` to."""
    start = time.perf_counter()
    holder: dict = {"output": None}
    cm, obs = _lf_open(name, input_data)
    try:
        yield holder
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        _write_local({"name": name, "kind": kind, "latency_ms": elapsed_ms,
                      "input": _trunc(input_data), "output": _trunc(holder.get("output"))})
        if obs is not None:
            try:
                obs.update(output=holder.get("output"))
            except Exception:  # noqa: BLE001
                pass
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass


def flush() -> None:
    if _langfuse:
        try:
            _langfuse.flush()
        except Exception:  # noqa: BLE001
            pass


def recent_traces(limit: int = 40) -> list[dict]:
    if not _TRACE_FILE.exists():
        return []
    out = []
    for ln in _TRACE_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(ln))
        except Exception:  # noqa: BLE001
            pass
    return out


def clear_traces() -> None:
    if _TRACE_FILE.exists():
        _TRACE_FILE.unlink()


def backend() -> str:
    return "langfuse" if _langfuse else "local-jsonl"


def _trunc(v, n: int = 600):
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"
