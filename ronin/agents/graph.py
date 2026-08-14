"""Supervisor graph (LangGraph).

A supervisor classifies intent and routes to one specialist sub-agent:
  • knowledge — cited RAG over internal docs
  • intel     — competitor price/BSR analysis
  • report    — daily Slack Block Kit brief
This is the multi-agent orchestration pattern (Susan → specialists) in miniature.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from ..llm import get_llm
from ..tracing import span, flush
from .state import AgentState
from .knowledge import knowledge_node
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


def supervisor(state: AgentState) -> dict:
    q = state["question"]
    with span("supervisor.route", q) as sp:
        route = _route(q)
        sp["output"] = route
    return {"route": route, "steps": [f"supervisor → {route}"]}


def _selector(state: AgentState) -> str:
    return state["route"]


def build_graph():
    b = StateGraph(AgentState)
    b.add_node("supervisor", supervisor)
    b.add_node("knowledge", knowledge_node)
    b.add_node("intel", intel_node)
    b.add_node("report", report_node)
    b.add_edge(START, "supervisor")
    b.add_conditional_edges("supervisor", _selector,
                            {"knowledge": "knowledge", "intel": "intel", "report": "report"})
    b.add_edge("knowledge", END)
    b.add_edge("intel", END)
    b.add_edge("report", END)
    return b.compile()


_GRAPH = None


def run(question: str) -> dict:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    with span("graph.invoke", question, kind="trace"):
        result = _GRAPH.invoke({"question": question})
    flush()  # push spans to Langfuse promptly (no-op when using local traces)
    return result
