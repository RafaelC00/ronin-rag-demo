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
# or eval in terminal:  python -m ronin.eval.run_eval
```

Runs with **only an OpenRouter key** (auto-read from `~/.config/openrouter/key`). No Docker.

---

## Architecture

```
UI (Streamlit)  ─┐
API (FastAPI) ───┤→  Supervisor (LangGraph)  ──►  Knowledge agent  (agentic/corrective RAG)
                 │                              ├►  Intel agent      (competitor deltas + analysis)
                 │                              └►  Report agent     (daily brief → Slack Block Kit)
                 │
 Retrieval:  hybrid (dense Qdrant + BM25) → RRF → rerank (VoyageAI / cross-encoder) → cited generation
 Observability:  Langfuse (or local JSONL trace)      Eval:  LLM-judge + deterministic (RAGAS-style)
```

### Why these choices (2026 practice)
- **Agentic / corrective RAG** — the knowledge agent grades its own retrieval and rewrites the
  query when context is weak, instead of answering from bad context.
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

## Stack

| Area | Technology |
|---|---|
| Orchestration | LangGraph / LangChain (supervisor + sub-agents, `ronin/agents/`) |
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
  agents/        supervisor + knowledge / intel / report (LangGraph)
  eval/          golden set + LLM-judge harness
  api.py         FastAPI
app.py           Streamlit UI
```
