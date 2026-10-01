"""FastAPI service exposing the agent fleet.

Sync path-operations run in FastAPI's threadpool, so blocking LLM calls don't
stall the event loop. In prod this containerizes onto ECS/Lambda behind an ALB.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .agents.graph import run as run_graph, resume as resume_graph
from .agents.report import report_node
from .tracing import recent_traces, backend
from .embeddings import get_embedder
from .config import get_settings

app = FastAPI(title="Ronin Athletics — Agentic RAG", version="0.1.0")


class ChatIn(BaseModel):
    question: str
    thread_id: str | None = None   # reuse to continue a checkpointed thread


class ResumeIn(BaseModel):
    thread_id: str
    approved: bool


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "llm": f"{s.ronin_llm_provider}:{s.ronin_llm_model}",
        "embeddings": get_embedder().label,
        "tracing": backend(),
    }


@app.post("/chat")
def chat(body: ChatIn):
    return _chat_payload(run_graph(body.question, body.thread_id))


@app.post("/chat/resume")
def chat_resume(body: ResumeIn):
    """Resolve the human confirmation gate on a paused thread (e.g. the daily brief)."""
    return _chat_payload(resume_graph(body.thread_id, body.approved))


def _chat_payload(state: dict) -> dict:
    return {
        "thread_id": state["thread_id"],
        "pending": state.get("pending"),
        "route": state.get("route"),
        "answer": state.get("answer"),
        "citations": state.get("citations", []),
        "steps": state.get("steps", []),
        "passages": state.get("passages", []),
    }


@app.post("/report/run")
def report():
    state = report_node({"question": "daily brief"})
    return {"text": state["report_text"], "blocks": state["report_blocks"], "steps": state["steps"]}


@app.get("/traces")
def traces(limit: int = 40):
    return {"backend": backend(), "traces": recent_traces(limit)}
