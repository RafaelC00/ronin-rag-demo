"""RAG evaluation harness (LLM-as-judge + deterministic checks).

Metrics (RAGAS-style, dependency-light so it runs anywhere):
  • context_precision : fraction of retrieved passages that are relevant  (LLM judge)
  • faithfulness      : is the answer grounded in the retrieved context?   (LLM judge)
  • answer_relevance  : does the answer address the question?              (LLM judge)
  • retrieval_hit     : did we retrieve at least one expected source?      (deterministic)
  • fact_recall       : did required facts appear in the answer?           (deterministic)

Swappable for the `ragas` library in prod — same metric surface.
"""
from __future__ import annotations

import json
import statistics

from ..llm import get_llm
from ..retrieval import retrieve
from ..agents.knowledge import _generate
from .dataset import EVAL_SET

_JUDGE_SYS = (
    "You are a strict RAG evaluator. Score 0.0-1.0. "
    'Return JSON: {"context_precision": x, "faithfulness": x, "answer_relevance": x}. '
    "context_precision = fraction of context passages relevant to the question; "
    "faithfulness = is every claim in the answer supported by the context; "
    "answer_relevance = does the answer address the question."
)


def _judge(question: str, context: str, answer: str) -> dict:
    user = f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
    try:
        d = get_llm().json(_JUDGE_SYS, user)
        return {k: float(d.get(k, 0.0)) for k in ("context_precision", "faithfulness", "answer_relevance")}
    except Exception:  # noqa: BLE001
        return {"context_precision": 0.0, "faithfulness": 0.0, "answer_relevance": 0.0}


def evaluate() -> dict:
    rows = []
    for case in EVAL_SET:
        q = case["question"]
        passages = retrieve(q, top_k=4)
        ctx = "\n\n".join(p.text for p in passages)
        answer = _generate(q, passages)

        got_sources = {p.source for p in passages}
        hit = any(s in got_sources for s in case["expected_sources"])
        recall = sum(1 for m in case["must_include"] if m.lower() in answer.lower()) / len(case["must_include"])
        judged = _judge(q, ctx, answer)

        rows.append({"question": q, "retrieval_hit": hit, "fact_recall": round(recall, 2), **judged})

    def avg(k):
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        return round(statistics.mean(vals), 3) if vals else 0.0

    summary = {
        "n": len(rows),
        "retrieval_hit_rate": round(sum(1 for r in rows if r["retrieval_hit"]) / len(rows), 3),
        "fact_recall": avg("fact_recall"),
        "context_precision": avg("context_precision"),
        "faithfulness": avg("faithfulness"),
        "answer_relevance": avg("answer_relevance"),
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result["summary"], indent=2))
    for r in result["rows"]:
        print(f"  hit={r['retrieval_hit']!s:5} recall={r['fact_recall']:.2f} "
              f"faith={r['faithfulness']:.2f} rel={r['answer_relevance']:.2f}  {r['question'][:60]}")
