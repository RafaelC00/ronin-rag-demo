# 🥋 Ronin Athletics — Agentic RAG + Multi-Agent Ops Platform

A portfolio demo: an **agentic RAG** system and a **fleet of collaborating agents** for a #1-on-Amazon BJJ / martial-arts apparel brand. It answers staff questions over an internal knowledge base (catalog, pricing, margins, policies, SOPs), tracks Amazon competitors, and auto-composes a **Slack Block Kit** daily brief — with full **evaluation**
and **observability**.

---

## Quickstart

```bash
cd ronin-rag-demo
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on mac/linux)
pip install -r requirements.txt

# (optional) cp .env.example .env  — runs out-of-the-box with your OpenRouter key
python -m ronin.ingest             # build the vector index (first run downloads a small embed model)
streamlit run app.py               # open the demo UI
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

## Maps 1:1 to the target JD stack

| JD requirement | In this demo |
|---|---|
| LangGraph / LangChain | supervisor + sub-agent graph (`ronin/agents/`) |
| RAG architecture, vector search / embeddings | `ronin/retrieval.py`, `ronin/vectorstore.py` |
| VoyageAI | embeddings + reranker adapter (`ronin/embeddings.py`, `retrieval.py`) |
| Langfuse / observability | `ronin/tracing.py` |
| Prompt engineering + evaluation | `ronin/eval/` |
| FastAPI, Python | `ronin/api.py` |
| MongoDB / cloud vector store | Qdrant now → Atlas Vector Search in prod (same interface) |
| AWS Bedrock | `ronin/llm.py` provider adapter |

---

## 🎤 5-minute interview demo script

1. **Frame the business need (30s).** "An Amazon-first apparel brand drowning in manual reporting
   and tribal knowledge. I designed an agentic platform so any employee can query internal data,
   a fleet tracks competitors and ships a Slack brief, all observable and evaluated."
2. **Ask tab (90s).** Ask *"What's the MAP and margin on the Shogun gi?"* → grounded, cited answer.
   Open **Agent steps** → show supervisor→knowledge, retrieve→self-grade→generate.
3. **Daily Brief tab (60s).** Generate → show the **Slack Block Kit** preview + JSON. "Deterministic
   facts, LLM writes only the narrative — trustworthy numbers."
4. **Evaluation tab (60s).** Run → faithfulness / retrieval / precision metrics. "This is how I keep
   a RAG honest and catch regressions — eval-driven development."
5. **Architecture tab (60s).** Point at the JD map: LangGraph, VoyageAI, Langfuse, Bedrock-swap,
   Atlas-swap, ECS/Lambda scale story.

**One-liner:** *"This is the same pattern I've shipped in production — I rebuilt a clean-room
version so I could show you the architecture end-to-end, running."*

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
