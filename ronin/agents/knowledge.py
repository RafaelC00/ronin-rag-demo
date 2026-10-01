"""Knowledge agent = agentic (corrective) RAG.

Flow: retrieve -> self-grade relevance -> (rewrite + re-retrieve if weak) ->
generate a grounded, cited answer. This is CRAG in spirit: the agent judges its
own context and repairs a bad retrieval instead of blindly answering.
"""
from __future__ import annotations

from ..llm import get_llm
from ..retrieval import retrieve, format_context, Passage
from ..tracing import span
from .state import AgentState

_GRADER_SYS = (
    "You grade whether retrieved context is sufficient to answer a question about the "
    "Ronin Athletics business (catalog, pricing/margins, competitors, policies, SOPs). "
    'Return JSON: {"sufficient": true|false, "rewrite": "a better search query or empty string"}.'
)

_ANSWER_SYS = (
    "You are the Ronin Athletics internal knowledge assistant. Answer ONLY from the provided "
    "context. Be concise and specific (cite numbers). If the context lacks the answer, say so. "
    "End with a 'Sources:' line listing the bracketed citations you used."
)


def _grade(question: str, passages: list[Passage]) -> dict:
    ctx = format_context(passages)
    try:
        return get_llm().json(_GRADER_SYS, f"Question: {question}\n\nContext:\n{ctx}")
    except Exception:  # noqa: BLE001
        return {"sufficient": True, "rewrite": ""}


def _generate(question: str, passages: list[Passage]) -> str:
    ctx = format_context(passages)
    return get_llm().complete(_ANSWER_SYS, f"Question: {question}\n\nContext:\n{ctx}")


def knowledge_node(state: AgentState) -> dict:
    q = state["question"]
    steps = list(state.get("steps", []))

    with span("knowledge.retrieve", q) as sp:
        passages = retrieve(q, top_k=4)
        sp["output"] = [p.citation for p in passages]
    steps.append(f"retrieve → {len(passages)} passages")

    with span("knowledge.grade", q) as sp:
        grade = _grade(q, passages)
        sp["output"] = grade
    if not grade.get("sufficient", True) and grade.get("rewrite"):
        steps.append(f"self-correct: rewrite → '{grade['rewrite']}'")
        with span("knowledge.re_retrieve", grade["rewrite"]) as sp:
            passages = retrieve(grade["rewrite"], top_k=4)
            sp["output"] = [p.citation for p in passages]
    else:
        steps.append("grade: context sufficient")

    with span("knowledge.generate", q) as sp:
        answer = _generate(q, passages)
        sp["output"] = answer
    steps.append("generate cited answer")

    return {
        "answer": answer,
        "citations": [p.citation for p in passages],
        "passages": [{"citation": p.citation, "text": p.text, "source": p.source} for p in passages],
        "steps": steps,
    }
