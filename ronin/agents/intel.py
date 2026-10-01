"""Competitive-intelligence agent.

Pulls the competitor snapshot + our catalog, computes price/BSR deltas
deterministically (no hallucinated numbers), then asks the LLM for a short
analyst read + one recommended action grounded in those computed facts.
"""
from __future__ import annotations

import json

from ..config import DATA_DIR
from ..llm import get_llm
from ..tracing import span
from .state import AgentState

_ANALYST_SYS = (
    "You are a senior Amazon competitive analyst for Ronin Athletics. Using ONLY the "
    "computed facts provided, write 2-3 tight sentences of analysis and ONE recommended "
    "action. Follow SOP: defend price with coupons (<=15% off MAP), never break MAP. "
    "No invented numbers."
)


def _compute_facts() -> dict:
    catalog = json.loads((DATA_DIR / "catalog.json").read_text(encoding="utf-8"))
    comp = json.loads((DATA_DIR / "competitors.json").read_text(encoding="utf-8"))
    flagship = next(p for p in catalog["products"] if p["sku"] == "RA-GI-SHOGUN-A2-BLK")
    gi_rivals = [c for c in comp["competitors"] if "Gi" in c["flagship"]]
    facts = {
        "flagship": {"name": flagship["name"], "map": flagship["map_usd"], "bsr": flagship["amazon_bsr"],
                     "inventory": flagship["inventory_units"]},
        "rivals": [],
    }
    for c in comp["competitors"]:
        delta_vs_map = round(c["price_usd"] - flagship["map_usd"], 2) if "Gi" in c["flagship"] else None
        facts["rivals"].append({
            "brand": c["brand"], "flagship": c["flagship"], "price": c["price_usd"], "bsr": c["amazon_bsr"],
            "price_delta_vs_our_gi_map": delta_vs_map, "recent_move": c["recent_move"],
        })
    return facts


def intel_node(state: AgentState) -> dict:
    steps: list[str] = []  # new steps only (steps is an append reducer)
    with span("intel.compute_deltas") as sp:
        facts = _compute_facts()
        sp["output"] = facts
    steps.append("compute price/BSR deltas (deterministic)")

    with span("intel.analyze", facts) as sp:
        read = get_llm().complete(_ANALYST_SYS, "Computed facts:\n" + json.dumps(facts, indent=2))
        sp["output"] = read
    steps.append("LLM analyst read + action")

    return {"intel": {"facts": facts, "analysis": read},
            "answer": read, "steps": steps}
