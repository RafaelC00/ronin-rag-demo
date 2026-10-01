"""RAG evaluation harness (LLM-as-judge + deterministic checks).

Metrics (RAGAS-style, dependency-light so it runs anywhere):
  • context_precision : fraction of retrieved passages that are relevant  (LLM judge)
  • faithfulness      : is the answer grounded in the retrieved context?   (LLM judge)
  • answer_relevance  : does the answer address the question?              (LLM judge)
  • retrieval_hit     : did we retrieve at least one expected source?      (deterministic)
  • fact_recall       : did required facts appear in the answer?           (deterministic)

Swappable for the `ragas` library in prod — same metric surface.

Two answer paths are scored on the same golden set: `baseline` (retrieve once,
generate) and `loop` (the corrective-RAG graph). Run:
    python -m ronin.eval.run_eval --mode baseline|loop [--out result.json]
"""
from __future__ import annotations

import json
import statistics
import time

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


def _answer_baseline(q: str) -> dict:
    """Single-pass RAG: retrieve once, generate. No grading, no re-query."""
    passages = retrieve(q, top_k=4)
    return {"passages": passages, "answer": _generate(q, passages), "retrievals": 1, "rewrites": 0}


def _answer_loop(q: str) -> dict:
    """Corrective-RAG graph: retrieve -> grade -> (rewrite -> retrieve)* -> generate."""
    from ..agents.knowledge import build_knowledge_graph
    from ..retrieval import Passage

    state = build_knowledge_graph().invoke({"question": q})
    passages = [Passage(id=p["id"], text=p["text"], source=p["source"], title=p["title"])
                for p in state["passages"]]
    return {"passages": passages, "answer": state["answer"],
            "retrievals": state.get("attempts", 1), "rewrites": state.get("attempts", 1) - 1}


MODES = {"baseline": _answer_baseline, "loop": _answer_loop}


def evaluate(mode: str = "baseline", limit: int | None = None) -> dict:
    # Batch-embed every golden question up front: one embeddings request
    # instead of one per case (the free Voyage tier allows 3 requests/min).
    from ..embeddings import get_embedder

    cases = EVAL_SET[:limit] if limit else EVAL_SET
    get_embedder().embed_queries([case["question"] for case in cases])
    answer_fn = MODES[mode]
    usage0 = get_llm().usage()

    rows = []
    for case in cases:
        q = case["question"]
        t0 = time.perf_counter()
        out = answer_fn(q)
        latency = round(time.perf_counter() - t0, 2)
        passages, answer = out["passages"], out["answer"]
        ctx = "\n\n".join(p.text for p in passages)

        # hit / recall are only defined for questions the KB can answer
        hit = recall = None
        if case["expected_sources"]:
            got_sources = {p.source for p in passages}
            hit = any(s in got_sources for s in case["expected_sources"])
        if case["must_include"]:
            recall = round(sum(1 for m in case["must_include"] if m.lower() in answer.lower())
                           / len(case["must_include"]), 2)
        judged = _judge(q, ctx, answer)

        rows.append({"question": q, "answerable": bool(case["expected_sources"]),
                     "retrieval_hit": hit, "fact_recall": recall, **judged,
                     "retrievals": out["retrievals"], "rewrites": out["rewrites"],
                     "latency_s": latency, "answer": answer})

    def avg(k, subset=None):
        vals = [r[k] for r in (subset or rows) if isinstance(r[k], (int, float)) and not isinstance(r[k], bool)]
        return round(statistics.mean(vals), 3) if vals else 0.0

    hits = [r for r in rows if r["retrieval_hit"] is not None]
    usage1 = get_llm().usage()
    summary = {
        "mode": mode,
        "n": len(rows),
        "retrieval_hit_rate": round(sum(1 for r in hits if r["retrieval_hit"]) / len(hits), 3) if hits else 0.0,
        "fact_recall": avg("fact_recall"),
        "context_precision": avg("context_precision"),
        "faithfulness": avg("faithfulness"),
        "answer_relevance": avg("answer_relevance"),
        "avg_retrievals": avg("retrievals"),
        "avg_latency_s": avg("latency_s"),
        "cases_that_rewrote": sum(1 for r in rows if r["rewrites"]),
        "llm_calls": usage1["calls"] - usage0["calls"],
        "est_cost_usd": round(usage1["est_cost_usd"] - usage0["est_cost_usd"], 4),
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score the golden set.")
    ap.add_argument("--mode", choices=sorted(MODES), default="baseline")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases")
    ap.add_argument("--out", default=None, help="write the full result (incl. answers) as JSON")
    args = ap.parse_args()

    result = evaluate(args.mode, args.limit)
    print(json.dumps(result["summary"], indent=2))
    for r in result["rows"]:
        rec = "  -- " if r["fact_recall"] is None else f"{r['fact_recall']:.2f}"
        print(f"  tries={r['retrievals']} recall={rec} faith={r['faithfulness']:.2f} "
              f"prec={r['context_precision']:.2f} rel={r['answer_relevance']:.2f}  {r['question'][:56]}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
