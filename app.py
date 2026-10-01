"""Ronin Athletics — UI (Streamlit).

Run:  streamlit run app.py
Tabs: Ask (agentic RAG) · Daily Brief (Slack Block Kit) · Evaluation · Traces · Architecture
"""
from __future__ import annotations

import os

import streamlit as st

# Streamlit Cloud exposes secrets via st.secrets, not the process env.
# Mirror them into os.environ BEFORE the ronin modules read their keys.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass  # no secrets file (local dev uses ~/.config fallbacks)

from ronin.config import get_settings
from ronin.embeddings import get_embedder
from ronin.agents.graph import run as run_graph
from ronin.agents.report import report_node
from ronin.slack import to_json
from ronin.tracing import recent_traces, backend, clear_traces

st.set_page_config(page_title="Ronin Athletics — Agentic RAG", page_icon="🥋", layout="wide")
s = get_settings()

with st.sidebar:
    st.title("🥋 Ronin Athletics")
    st.caption("Agentic RAG + multi-agent ops platform")
    st.markdown("**Live config**")
    st.code(
        f"LLM        : {s.ronin_llm_provider}:{s.ronin_llm_model}\n"
        f"Embeddings : {get_embedder().label}\n"
        f"Vector DB  : Qdrant (embedded)\n"
        f"Retrieval  : hybrid (dense+BM25) → rerank\n"
        f"Tracing    : {backend()}",
        language="text",
    )
    st.markdown("**Try asking**")
    st.markdown(
        "- What's the MAP and margin on the Shogun gi?\n"
        "- How deep can we discount below MAP?\n"
        "- Which competitor undercut our gi price?\n"
        "- Run the daily brief")

tab_ask, tab_brief, tab_eval, tab_trace, tab_arch = st.tabs(
    ["💬 Ask", "📊 Daily Brief", "🧪 Evaluation", "🔎 Traces", "🏗️ Architecture"])

with tab_ask:
    st.subheader("Ask the agent fleet")
    q = st.text_input("Question", "What's the MAP price and gross margin on the flagship Shogun gi?")
    if st.button("Run", type="primary"):
        with st.spinner("supervisor → routing → retrieving → generating…"):
            state = run_graph(q)
        st.success(f"Routed to **{state.get('route')}** agent")
        st.markdown(state.get("answer", ""))
        cites = state.get("citations", [])
        if cites:
            st.caption("Sources: " + " ".join(dict.fromkeys(cites)))
        with st.expander("Agent steps (multi-agent trace)"):
            for step in state.get("steps", []):
                st.markdown(f"- {step}")
        passages = state.get("passages", [])
        if passages:
            with st.expander(f"Retrieved context ({len(passages)} passages)"):
                for p in passages:
                    st.markdown(f"**{p['citation']}**\n\n{p['text']}")

with tab_brief:
    st.subheader("Daily Amazon Brief → Slack Block Kit")
    st.caption("The report agent computes facts deterministically, writes the narrative, and formats Block Kit.")
    if st.button("Generate brief", type="primary"):
        with st.spinner("gathering facts → writing → building Block Kit…"):
            state = report_node({"question": "daily brief"})
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("#### Slack preview")
            st.info(state["report_text"])
        with col2:
            st.markdown("#### Block Kit JSON")
            st.code(to_json(state["report_blocks"]), language="json")
        with st.expander("Agent steps"):
            for step in state["steps"]:
                st.markdown(f"- {step}")

with tab_eval:
    st.subheader("RAG evaluation (eval-driven development)")
    st.caption("LLM-as-judge + deterministic checks over a golden set. RAGAS-compatible metric surface.")
    if st.button("Run evaluation", type="primary"):
        from ronin.eval.run_eval import evaluate
        try:
            with st.spinner("scoring the golden set… (~1 min: 6 cases × LLM judge)"):
                result = evaluate()
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "RateLimitError":
                st.warning(
                    "The embeddings provider (free tier) is rate-limited right now. "
                    "Wait ~1 minute and run again — results are cached after the first pass."
                )
                st.stop()
            raise
        sm = result["summary"]
        c = st.columns(5)
        c[0].metric("Retrieval hit", f"{sm['retrieval_hit_rate']*100:.0f}%")
        c[1].metric("Fact recall", f"{sm['fact_recall']*100:.0f}%")
        c[2].metric("Context precision", f"{sm['context_precision']*100:.0f}%")
        c[3].metric("Faithfulness", f"{sm['faithfulness']*100:.0f}%")
        c[4].metric("Answer relevance", f"{sm['answer_relevance']*100:.0f}%")
        st.dataframe(result["rows"], use_container_width=True)

with tab_trace:
    st.subheader(f"Traces — backend: {backend()}")
    cc = st.columns([1, 1, 4])
    if cc[0].button("Refresh"):
        st.rerun()
    if cc[1].button("Clear"):
        clear_traces(); st.rerun()
    for t in reversed(recent_traces(60)):
        st.markdown(f"**{t['name']}** · `{t['latency_ms']} ms`")
        if t.get("output"):
            st.caption(str(t["output"])[:400])

with tab_arch:
    st.subheader("Architecture")
    st.markdown(
        """
**Supervisor (LangGraph)** routes each request to one specialist sub-agent:

| Agent | Job | Technique |
|---|---|---|
| **Knowledge** | cited answers over internal KB | agentic/corrective RAG (retrieve → self-grade → rewrite → generate) |
| **Intel** | competitor price/BSR analysis | deterministic deltas + LLM analyst read |
| **Report** | daily brief → Slack | computed facts + LLM narrative + Block Kit |

**Retrieval:** hybrid (dense Qdrant + BM25) → Reciprocal Rank Fusion → reranker (VoyageAI / cross-encoder) → grounded, cited generation.

**Portability:** LLM provider adapter (OpenRouter ↔ **AWS Bedrock** via one env var); vector store is drop-in **MongoDB Atlas Vector Search** in prod; **Langfuse** tracing; **RAGAS**-style eval.

**Scale story:** async FastAPI on ECS/Lambda, batched embeddings, semantic cache, stateless agents + LangGraph checkpointer, horizontally-scaled vector DB.
        """)
    st.caption("Built with LangGraph, LangChain, VoyageAI, Langfuse, FastAPI, Qdrant, with MongoDB Atlas and Bedrock as drop-in production targets.")
