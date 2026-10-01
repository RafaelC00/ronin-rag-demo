"""Knowledge agent = corrective RAG as a real graph cycle.

    retrieve --> grade --+--> generate                (context supports an answer)
        ^                |
        +---- rewrite <--+                            (context is weak, budget left)

The grader decides whether the retrieved context actually supports answering.
A weak verdict sends the graph through `rewrite` and back into `retrieve`; an
explicit step budget (`budget`, default `RONIN_MAX_RETRIEVALS`) bounds the loop,
and the loop also stops if the rewriter has nothing new to try. Whatever the
reason it stops, `generate` still runs, and it is told to refuse when the
context lacks the answer.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from ..config import get_settings
from ..llm import get_llm
from ..retrieval import retrieve, format_context, Passage
from ..tracing import span
from .state import AgentState

TOP_K = 4
MAX_CONTEXT = 6   # passages carried across attempts (new hits first, then earlier ones)

_GRADER_SYS = (
    "You grade whether retrieved context is sufficient to answer a question about the "
    "Ronin Athletics business (catalog, pricing/margins, competitors, policies, SOPs). "
    "Sufficient means the context explicitly contains every fact needed to answer every part "
    "of the question. Do not use outside knowledge. "
    'Return JSON: {"sufficient": true|false, "missing": "what is missing, or empty", "reason": "one short sentence"}.'
)

_REWRITE_SYS = (
    "You rewrite a search query for a hybrid (semantic + keyword) index over Ronin Athletics "
    "internal documents (catalog, pricing, competitors, policies, SOPs, brand). "
    "The earlier queries did not surface what is needed. Write ONE new query that targets the "
    "missing information with different, more specific wording (use document vocabulary such as "
    "policy names, SOP topics, 'MAP', 'margin', 'coupon', 'reorder'). Do not repeat an earlier query. "
    'Return JSON: {"query": "..."}.'
)

_ANSWER_SYS = (
    "You are the Ronin Athletics internal knowledge assistant. Answer ONLY from the provided "
    "context. Be concise and specific (cite numbers). If the context lacks the answer, say so. "
    "End with a 'Sources:' line listing the bracketed citations you used."
)


def _as_dict(p: Passage) -> dict:
    return {"id": p.id, "citation": p.citation, "text": p.text, "source": p.source, "title": p.title}


def _as_passage(d: dict) -> Passage:
    return Passage(id=d["id"], text=d["text"], source=d["source"], title=d["title"])


def _generate(question: str, passages: list[Passage]) -> str:
    ctx = format_context(passages)
    return get_llm().complete(_ANSWER_SYS, f"Question: {question}\n\nContext:\n{ctx}")


# ---------------------------------------------------------------- nodes

def retrieve_node(state: AgentState) -> dict:
    query = state.get("query") or state["question"]
    attempts = state.get("attempts", 0) + 1
    budget = state.get("budget") or get_settings().ronin_max_retrievals

    with span("knowledge.retrieve", query) as sp:
        found = [_as_dict(p) for p in retrieve(query, top_k=TOP_K)]
        sp["output"] = [p["citation"] for p in found]

    # keep earlier hits: a corrective query often finds the missing piece, not the whole answer
    seen = {p["id"] for p in found}
    merged = found + [p for p in state.get("passages", []) if p["id"] not in seen]
    merged = merged[:MAX_CONTEXT]

    return {
        "query": query, "queries": [*state.get("queries", []), query],
        "attempts": attempts, "budget": budget, "passages": merged,
        "steps": [f"retrieve #{attempts}/{budget} '{query}' → {len(found)} passages"],
    }


def grade_node(state: AgentState) -> dict:
    q = state["question"]
    ctx = format_context([_as_passage(p) for p in state["passages"]])
    with span("knowledge.grade", q) as sp:
        try:
            raw = get_llm().json(_GRADER_SYS, f"Question: {q}\n\nContext:\n{ctx}")
            grade = {"sufficient": bool(raw.get("sufficient", True)),
                     "missing": str(raw.get("missing") or ""), "reason": str(raw.get("reason") or "")}
            error = False
        except Exception:  # noqa: BLE001 — a broken grader must not block an answer
            grade = {"sufficient": True, "missing": "", "reason": "grader unavailable"}
            error = True
        sp["output"] = grade
    out: dict = {"grade": grade}
    if error:
        out["stop_reason"] = "grader_error"
        out["steps"] = ["grade: grader unavailable → answering with what we have"]
    else:
        verdict = "sufficient" if grade["sufficient"] else f"weak ({grade['missing'] or grade['reason']})"
        out["steps"] = [f"grade: {verdict}"]
    return out


def rewrite_node(state: AgentState) -> dict:
    q = state["question"]
    tried = state.get("queries", [])
    grade = state.get("grade", {})
    user = (f"Question: {q}\nEarlier queries: {tried}\n"
            f"What was missing: {grade.get('missing') or grade.get('reason') or 'unknown'}")
    with span("knowledge.rewrite", user) as sp:
        try:
            new_q = str(get_llm().json(_REWRITE_SYS, user).get("query") or "").strip()
        except Exception:  # noqa: BLE001
            new_q = ""
        sp["output"] = new_q
    if not new_q or new_q.lower() in {t.lower() for t in tried}:
        return {"stop_reason": "no_new_query", "steps": ["rewrite: no new query to try → stop"]}
    return {"query": new_q, "steps": [f"rewrite → '{new_q}'"]}


def generate_node(state: AgentState) -> dict:
    q = state["question"]
    passages = [_as_passage(p) for p in state["passages"]]
    reason = state.get("stop_reason")
    if not reason:
        reason = "sufficient" if state.get("grade", {}).get("sufficient", True) else "budget"
    with span("knowledge.generate", q) as sp:
        answer = _generate(q, passages)
        sp["output"] = answer
    label = {"sufficient": "context sufficient", "budget": "step budget spent",
             "no_new_query": "no new query", "grader_error": "grader unavailable"}[reason]
    return {
        "answer": answer, "stop_reason": reason,
        "citations": [p.citation for p in passages],
        "steps": [f"generate cited answer ({label}, {state['attempts']} retrieval(s))"],
    }


# ---------------------------------------------------------------- wiring

def after_grade(state: AgentState) -> str:
    """The cycle's exit test: answer, or go around again if the budget allows."""
    if state.get("stop_reason") or state["grade"]["sufficient"]:
        return "generate"
    if state["attempts"] >= state["budget"]:
        return "generate"
    return "rewrite"


def after_rewrite(state: AgentState) -> str:
    return "generate" if state.get("stop_reason") == "no_new_query" else "retrieve"


def add_corrective_rag(b: StateGraph) -> None:
    """Add the retrieve/grade/rewrite/generate nodes and their edges. Entry: `retrieve`. Exit: `generate`."""
    b.add_node("retrieve", retrieve_node)
    b.add_node("grade", grade_node)
    b.add_node("rewrite", rewrite_node)
    b.add_node("generate", generate_node)
    b.add_edge("retrieve", "grade")
    b.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})
    b.add_conditional_edges("rewrite", after_rewrite, {"retrieve": "retrieve", "generate": "generate"})


def build_knowledge_graph(checkpointer=None):
    """The corrective-RAG cycle on its own (used by the eval harness)."""
    b = StateGraph(AgentState)
    add_corrective_rag(b)
    b.add_edge(START, "retrieve")
    b.add_edge("generate", END)
    return b.compile(checkpointer=checkpointer)
