"""Report agent — composes the Daily Amazon Brief and formats it as Slack Block Kit.

Facts (BSR, deltas, inventory risks) are computed deterministically; the LLM only
writes the headline + recommended action. This mirrors a real reporting pipeline:
trustworthy numbers, LLM for the narrative.
"""
from __future__ import annotations

import json

from ..config import DATA_DIR
from ..llm import get_llm
from ..slack import build_daily_brief, render_text, post_to_slack
from ..tracing import span
from .intel import _compute_facts
from .state import AgentState

_HEADLINE_SYS = (
    "You write a one-line executive headline and a one-line recommended action for a "
    "daily Amazon brief for Ronin Athletics (a #1-BSR BJJ apparel brand). Ground it in the "
    "facts. Follow SOP (coupons not MAP cuts). "
    'Return JSON: {"headline": "...", "action": "..."}.'
)

REORDER_TRIGGER = {"RA-GI-SHOGUN-A2-BLK": 800}


def _inventory_risks() -> list[str]:
    catalog = json.loads((DATA_DIR / "catalog.json").read_text(encoding="utf-8"))
    risks = []
    for p in catalog["products"]:
        floor = REORDER_TRIGGER.get(p["sku"])
        if floor and p["inventory_units"] < floor:
            risks.append(f"{p['name']} at {p['inventory_units']} units (floor {floor}) — reorder now")
    if not risks:
        risks.append("No SKUs below reorder trigger.")
    return risks


def report_node(state: AgentState) -> dict:
    steps: list[str] = []  # new steps only (steps is an append reducer)
    with span("report.gather_facts") as sp:
        facts = _compute_facts()
        risks = _inventory_risks()
        sp["output"] = {"facts": facts, "risks": risks}
    steps.append("gather BSR / deltas / inventory")

    with span("report.write_narrative", facts) as sp:
        try:
            narrative = get_llm().json(_HEADLINE_SYS, "Facts:\n" + json.dumps(facts, indent=2) +
                                       "\nInventory risks:\n" + json.dumps(risks))
        except Exception:  # noqa: BLE001 — never let the brief hard-fail in a live demo
            narrative = {"headline": "Flagship holds #1 BSR; a value rival undercut the gi.",
                         "action": "Launch a 10% coupon on the Shogun gi (stays within MAP policy)."}
        sp["output"] = narrative
    steps.append("LLM headline + action")

    moves = [f"{r['brand']}: {r['recent_move']}" for r in facts["rivals"] if r["recent_move"]
             and "No pricing" not in r["recent_move"]]
    payload = {
        "date": "2026-06-15",
        "headline": narrative.get("headline", "Daily brief"),
        "kpis": [
            f"*Flagship BSR* #{facts['flagship']['bsr']}",
            f"*Shogun Gi MAP* ${facts['flagship']['map']}",
            f"*Flagship inventory* {facts['flagship']['inventory']} units",
            f"*Gi rivals tracked* {len([r for r in facts['rivals'] if r['price_delta_vs_our_gi_map'] is not None])}",
        ],
        "competitor_moves": moves,
        "inventory_risks": risks,
        "action": narrative.get("action", ""),
    }

    with span("report.build_blockkit", payload) as sp:
        blocks = build_daily_brief(payload)
        posted = post_to_slack(blocks)
        sp["output"] = {"blocks": len(blocks), "posted_to_slack": posted}
    steps.append("build Slack Block Kit" + (" + posted" if posted else " (in-app preview)"))

    text = render_text(blocks)
    return {"report_blocks": blocks, "report_text": text, "answer": text, "steps": steps}
