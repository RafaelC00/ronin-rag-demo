"""Supervisor graph (LangGraph).

A supervisor classifies intent and routes to one specialist:

    START -> supervisor -+-> retrieve -> grade -+-> generate -> END      knowledge (corrective RAG)
                         |       ^              |
                         |       +-- rewrite <--+                         (cycle, bounded by a step budget)
                         +-> intel -> END
                         +-> confirm_report -+-> report -> END           human confirms first
                                             +-> cancelled -> END        (interrupt; resumable)

State is checkpointed to SQLite, so a thread (including one paused at the
confirmation gate) can be resumed from a new process.
"""
from __future__ import annotations

import sqlite3
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from ..config import CHECKPOINT_PATH, get_settings
from ..llm import get_llm
from ..tracing import span, flush
from .state import AgentState
from .knowledge import add_corrective_rag
from .intel import intel_node
from .report import report_node

_ROUTER_SYS = (
    "Route a user request for the Ronin Athletics AI platform to exactly one agent:\n"
    "- 'report'    : wants the daily brief / Slack report / morning summary\n"
    "- 'intel'     : asks about competitors, competitor pricing, market moves, threats\n"
    "- 'knowledge' : anything about our own products, prices, margins, policies, SOPs, inventory\n"
    'Return JSON: {"route": "knowledge|intel|report"}.'
)


def _route(question: str) -> str:
    q = question.lower()
    # cheap deterministic hints first (fast + robust for the demo)
    if any(w in q for w in ("daily brief", "report", "slack", "morning", "summary")):
        return "report"
    if any(w in q for w in ("competitor", "competition", "rival", "osoto", "guard pass", "submission co", "ironclad", "market")):
        return "intel"
    try:
        r = get_llm().json(_ROUTER_SYS, question).get("route", "knowledge")
        return r if r in ("knowledge", "intel", "report") else "knowledge"
    except Exception:  # noqa: BLE001
        return "knowledge"


# per-turn fields cleared at the start of every question, so a reused thread starts clean
_RESET = {"query": "", "queries": [], "attempts": 0, "budget": 0, "passages": [], "grade": {},
          "stop_reason": "", "answer": "", "citations": [], "approved": False}


def supervisor(state: AgentState) -> dict:
    q = state["question"]
    with span("supervisor.route", q) as sp:
        route = _route(q)
        sp["output"] = route
    return {**_RESET, "route": route, "steps": [f"supervisor → {route}"]}


def _selector(state: AgentState) -> str:
    # `knowledge` enters the corrective-RAG cycle at its first node
    return {"knowledge": "retrieve", "intel": "intel", "report": "confirm_report"}[state["route"]]


def _needs_confirmation() -> bool:
    s = get_settings()
    if s.ronin_confirm_report == "always":
        return True
    if s.ronin_confirm_report == "never":
        return False
    return bool(s.slack_webhook_url)   # auto: only when the brief would be posted outward


def confirm_report(state: AgentState) -> dict:
    """Human-in-the-loop gate before the report agent runs.

    The report costs LLM calls and, when a Slack webhook is configured, posts to a
    channel. `interrupt()` pauses the graph here and persists the thread; it resumes
    with the human's decision as the return value. Nothing with side effects happens
    in this node before the interrupt, because the node re-runs from the top on resume.
    """
    if not _needs_confirmation():
        return {"approved": True, "steps": ["report gate: confirmation not required"]}
    posts = bool(get_settings().slack_webhook_url)
    decision = interrupt({
        "action": "daily_brief",
        "question": "Generate the daily brief" + (" and post it to Slack?" if posts else "?"),
        "posts_to_slack": posts,
    })
    approved = bool(decision)
    return {"approved": approved,
            "steps": ["report gate: approved by human" if approved else "report gate: declined by human"]}


def _after_gate(state: AgentState) -> str:
    return "report" if state.get("approved") else "cancelled"


def cancelled(state: AgentState) -> dict:
    return {"answer": "Report cancelled. Nothing was generated or posted.", "steps": ["report cancelled"]}


def build_graph(checkpointer=None):
    b = StateGraph(AgentState)
    b.add_node("supervisor", supervisor)
    add_corrective_rag(b)                      # retrieve, grade, rewrite, generate (+ the cycle)
    b.add_node("intel", intel_node)
    b.add_node("confirm_report", confirm_report)
    b.add_node("report", report_node)
    b.add_node("cancelled", cancelled)
    b.add_edge(START, "supervisor")
    b.add_conditional_edges("supervisor", _selector,
                            {"retrieve": "retrieve", "intel": "intel", "confirm_report": "confirm_report"})
    b.add_edge("generate", END)
    b.add_edge("intel", END)
    b.add_conditional_edges("confirm_report", _after_gate, {"report": "report", "cancelled": "cancelled"})
    b.add_edge("report", END)
    b.add_edge("cancelled", END)
    return b.compile(checkpointer=checkpointer)


def make_checkpointer(path=None):
    """SQLite-backed checkpointer (no service, no account). Falls back to memory if unavailable."""
    path = path or CHECKPOINT_PATH
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Streamlit/FastAPI call from worker threads; SqliteSaver serialises writes
        return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    except Exception as e:  # noqa: BLE001
        from langgraph.checkpoint.memory import InMemorySaver
        print(f"[graph] SQLite checkpointer unavailable ({e}); threads will not survive a restart")
        return InMemorySaver()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(make_checkpointer())
    return _GRAPH


def _result(config, state: dict) -> dict:
    """Plain state dict plus thread bookkeeping; surfaces a pending confirmation as `pending`."""
    out = {k: v for k, v in state.items() if k != "__interrupt__"}
    out["thread_id"] = config["configurable"]["thread_id"]
    pending = state.get("__interrupt__")
    if pending:
        out["pending"] = pending[0].value
        out["steps"] = [*out.get("steps", []), "paused: waiting for human confirmation"]
    return out


def run(question: str, thread_id: str | None = None) -> dict:
    """Run a question. If the graph pauses at the confirmation gate the result has a `pending` key;
    call `resume(thread_id, approved)` to continue."""
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    with span("graph.invoke", question, kind="trace"):
        state = get_graph().invoke({"question": question}, config)
    flush()  # push spans to Langfuse promptly (no-op when using local traces)
    return _result(config, state)


def resume(thread_id: str, approved: bool) -> dict:
    """Continue a thread paused at the confirmation gate (works from a fresh process)."""
    config = {"configurable": {"thread_id": thread_id}}
    with span("graph.resume", thread_id, kind="trace"):
        state = get_graph().invoke(Command(resume=bool(approved)), config)
    flush()
    return _result(config, state)


def thread_state(thread_id: str) -> dict | None:
    """Saved state of a thread, or None if the thread is unknown."""
    snap = get_graph().get_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        return None
    out = dict(snap.values)
    out["thread_id"] = thread_id
    if snap.next:
        out["next"] = list(snap.next)
        for t in snap.tasks:
            if t.interrupts:
                out["pending"] = t.interrupts[0].value
    return out
