# Ronin Athletics — Agentic RAG + Multi-Agent Ops Platform

An **agentic RAG** system and a **fleet of collaborating agents** for a #1-on-Amazon BJJ / martial-arts apparel brand. It answers staff questions over an internal knowledge base (catalog, pricing, margins, policies, SOPs), tracks Amazon competitors, and auto-composes a **Slack Block Kit** daily brief — with full **evaluation**
and **observability**.

> Ronin Athletics is a fictional brand. All catalog, pricing, competitor, policy and SOP data in `data/` is synthetic.

---

## Quickstart

```bash
cd ronin-rag-demo
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on mac/linux)
pip install -r requirements.txt

# (optional) cp .env.example .env  — runs out-of-the-box with your OpenRouter key
python -m ronin.ingest             # build the vector index (first run downloads a small embed model)
streamlit run app.py               # open the UI
# or serve the API:  uvicorn ronin.api:app --reload
# or eval in terminal:  python -m ronin.eval.run_eval --mode loop   (or baseline)
# tests:  pip install -r requirements-dev.txt && pytest
```

Runs with **only an OpenRouter key** (auto-read from `~/.config/openrouter/key`). No Docker.

---

## Architecture

```
UI (Streamlit)  ─┐
API (FastAPI) ───┤→  Supervisor (LangGraph)  ──►  Knowledge agent  (corrective-RAG cycle)
                 │                              ├►  Intel agent      (competitor deltas + analysis)
                 │                              └►  Report agent     (confirm gate → daily brief → Slack Block Kit)
                 │
 Retrieval:  hybrid (dense Qdrant + BM25) → RRF → rerank (VoyageAI / cross-encoder) → cited generation
 Observability:  Langfuse (or local JSONL trace)      Eval:  LLM-judge + deterministic (RAGAS-style)
```

### Why these choices (2026 practice)
- **Agentic / corrective RAG** — the knowledge agent grades its own retrieval and, when context is
  weak, rewrites the query and retrieves again (a bounded graph cycle), instead of answering from bad context.
- **Hybrid retrieval + reranking** — dense semantics + BM25 exact-match (SKUs, "MAP"), fused with
  Reciprocal Rank Fusion, then a reranker sharpens top-k.
- **Grounded + cited** — answers cite `[source:section]`; the generator is instructed to refuse
  when context is insufficient.
- **Eval-driven** — a golden set scored on retrieval hit, fact recall, context precision,
  faithfulness, answer relevance.
- **Observable** — every agent step + LLM call is traced.
- **Portable** — LLM provider adapter flips OpenRouter → **AWS Bedrock** with one env var; the
  vector store is a drop-in swap to **MongoDB Atlas Vector Search**.

---

## The graph

```
START -> supervisor --+--> retrieve -> grade --+--> generate -> END     knowledge
                      |        ^               |                        (corrective RAG)
                      |        +--- rewrite <--+
                      |
                      +--> intel -> END
                      |
                      +--> confirm_report --+--> report -> END          report
                                            +--> cancelled -> END       (human-in-the-loop)
```

The supervisor routes as before (keyword hints, then an LLM router). What changed is what sits behind the
`knowledge` and `report` routes:

- **Corrective-RAG cycle.** `retrieve` runs hybrid search, `grade` asks an LLM whether the retrieved context
  explicitly contains every fact the question needs, and a weak verdict goes through `rewrite` (a new query
  aimed at what is missing, told which queries were already tried) and back into `retrieve`. Passages found in
  earlier attempts are kept, up to 6. The cycle ends when the grade is sufficient, **when the step budget is
  spent** (`RONIN_MAX_RETRIEVALS`, default 3 retrievals: the first plus up to two rewrites), or when the
  rewriter has nothing new to try. In every case `generate` still runs and refuses if the context does not
  contain the answer. If the grader itself fails, the graph answers with what it has.
- **Checkpointer.** Every thread is saved to a local SQLite file (`.state/checkpoints.sqlite`,
  `RONIN_CHECKPOINT_PATH` to move it) through `langgraph-checkpoint-sqlite`, so a thread can be resumed after a
  restart. No service, no account. If the package is missing it falls back to an in-memory saver.
- **Human-in-the-loop.** `confirm_report` calls LangGraph's `interrupt()` before the report agent runs, because
  that step spends LLM calls and posts to Slack when a webhook is set. The graph pauses, the thread is
  persisted, and `resume(thread_id, approved)` (or `POST /chat/resume`, or the Approve / Decline buttons in the
  UI) continues it, including from a new process. `RONIN_CONFIRM_REPORT=auto|always|never`; `auto` (default)
  only asks when a Slack webhook is configured.

```python
from ronin.agents.graph import run, resume
state = run("run the daily brief")           # -> state["pending"], state["thread_id"]
state = resume(state["thread_id"], True)     # same call works from another process
```

Before this change the graph was `START -> supervisor -> (knowledge | intel | report) -> END`: no cycle, no
checkpointer, no pause point. The knowledge node did one grade-and-maybe-rewrite pass inside a single function.

---

## Does the cycle help? Measured

`python -m ronin.eval.run_eval --mode baseline|loop` scores the same golden set (`ronin/eval/dataset.py`,
14 questions) through two pipelines: **baseline** (retrieve once, generate) and **loop** (the graph above).
Metrics are LLM-judged (context precision, faithfulness, answer relevance) plus deterministic checks
(retrieval hit, fact recall).

Setup: Claude Haiku 4.5 via OpenRouter for generation, grading, rewriting and judging; local fastembed
embeddings and cross-encoder reranker (so the numbers can be reproduced without an embeddings key);
2 runs per pipeline, mean shown. The 14 questions are the original 6 plus 8 added before the first run
(paraphrases, a multi-part question, and 2 the knowledge base cannot answer). Nothing in the set was changed
afterwards.

| | baseline | loop |
|---|---|---|
| Faithfulness | 0.995 | 0.994 |
| Answer relevance | 0.991 | 0.991 |
| Context precision | 0.846 | 0.850 |
| Retrieval hit rate | 1.000 | 1.000 |
| **Fact recall** (12 answerable questions) | **0.917** | **1.000** |
| Retrievals per question | 1.0 | 1.71 |
| Latency per question | 3.0 s | 8.3 s |
| LLM calls per run | 28 | 62 |
| Cost per run (est.) | $0.029 | $0.059 |

**Reading it honestly.**

- **Faithfulness did not move.** It was already ~1.0 without the loop: the generator is told to answer only
  from context and to refuse otherwise, and it does. There is no headroom on this set for the loop to show up there.
- **Fact recall did move, through one question.** "Which product line earns us the best margin at MAP?" was
  answered from a partial retrieval (it named the Core hoodie at 64.8% and missed the belt at 71.1%). The answer
  was faithful to the context it was given, so the faithfulness judge scored it 1.0 even though it was wrong.
  The loop re-queried, found the pricing table, and answered correctly in both runs. That is the whole
  recall gain: 1 question in 12, so it is a signal, not a statistic.
- **The loop fires more often than it should.** It rewrote on 5 of 14 questions. Two were the unanswerable
  ones (spending the full budget, then refusing, as intended); one was the margin question above (useful);
  two were answerable from the first retrieval (wasted: the grader was stricter than needed). On the
  multi-part SOP question the extra context also made the answer longer, and the judge scored faithfulness
  0.92 instead of 1.0.
- **Cost of the cycle:** about 1.7x the retrievals, 2.7x the latency and 2x the spend per question for the
  gain above. Whether that is worth it depends on how often questions need more than one retrieval; on this
  set, 1 in 12 did.

Caveats: 14 questions, 2 runs, one judge model that is also the generator (self-preference is possible), and
the local embedding stack rather than the hosted one the live demo uses. Faithfulness as measured cannot see
an answer that is faithful to incomplete context, which is the failure this loop exists to catch; fact recall
can, but only for questions that have a `must_include` list.

---

## Stack

| Area | Technology |
|---|---|
| Orchestration | LangGraph: supervisor, corrective-RAG cycle, SQLite checkpointer, `interrupt()` gate (`ronin/agents/`) |
| Retrieval | Hybrid dense + BM25 with RRF (`ronin/retrieval.py`, `ronin/vectorstore.py`) |
| Embeddings / rerank | VoyageAI or local fastembed (`ronin/embeddings.py`) |
| Vector store | Qdrant (embedded); interface compatible with MongoDB Atlas Vector Search |
| Observability | Langfuse or local JSONL trace (`ronin/tracing.py`) |
| Evaluation | Golden set + LLM-judge (`ronin/eval/`) |
| API / UI | FastAPI (`ronin/api.py`), Streamlit (`app.py`) |
| LLM providers | OpenRouter, Anthropic, AWS Bedrock (`ronin/llm.py`) |

---

## Layout
```
data/            synthetic Ronin KB (catalog, pricing, competitors, policies, SOPs, brand)
ronin/
  config.py      settings + graceful key resolution
  llm.py         provider adapter (OpenRouter | Anthropic | Bedrock)
  embeddings.py  VoyageAI | local fastembed
  vectorstore.py Qdrant embedded
  ingest.py      structure-aware chunking → embed → index
  retrieval.py   hybrid + RRF + rerank + citations
  slack.py       Block Kit builder / preview / webhook
  tracing.py     Langfuse | local trace
  agents/        supervisor + knowledge (corrective-RAG cycle) / intel / report (LangGraph)
  eval/          golden set + LLM-judge harness (baseline vs loop)
tests/           graph tests (fake LLM + retriever, offline)
  api.py         FastAPI
app.py           Streamlit UI
```
