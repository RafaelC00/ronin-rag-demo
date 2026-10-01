"""Ingestion: structured + semantic chunking -> embeddings -> Qdrant.

Chunking strategy (a 2026 best practice = structure-aware, not blind fixed-size):
  - catalog / pricing / competitors JSON  -> one record-level chunk each (clean metadata)
  - markdown docs (policies/sops/brand)    -> split on ## headings (semantic sections)
Each chunk keeps rich metadata (source, type, sku, title) for filtering + citations.
"""
from __future__ import annotations

import json
import re

from .config import DATA_DIR, CHUNKS_PATH
from .embeddings import get_embedder
from . import vectorstore as vs


def _md_sections(text: str, source: str) -> list[dict]:
    parts = re.split(r"\n(?=##\s)", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"#+\s*(.+)", part)
        title = m.group(1).strip() if m else source
        chunks.append({"source": source, "type": "doc", "title": title, "text": part})
    return chunks


def build_chunks() -> list[dict]:
    chunks: list[dict] = []

    catalog = json.loads((DATA_DIR / "catalog.json").read_text(encoding="utf-8"))
    for p in catalog["products"]:
        text = (
            f"Product: {p['name']} ({p['line']} line, {p['type']}). SKU {p['sku']}, ASIN {p['asin']}. "
            f"Color {p['color']}, size {p['size']}. MSRP ${p['msrp_usd']}, MAP ${p['map_usd']}, "
            f"unit cost ${p['unit_cost_usd']}. Amazon BSR #{p['amazon_bsr']}, rating {p['amazon_rating']} "
            f"from {p['amazon_reviews']} reviews. Inventory {p['inventory_units']} units. {p.get('notes','')}"
        )
        chunks.append({"source": "catalog", "type": "product", "title": p["name"], "sku": p["sku"], "text": text})

    pricing = json.loads((DATA_DIR / "pricing.json").read_text(encoding="utf-8"))
    for it in pricing["items"]:
        text = (
            f"Pricing/margin for SKU {it['sku']}: MSRP ${it['msrp']}, MAP ${it['map']}, "
            f"unit cost ${it['unit_cost']}, gross margin at MAP {it['margin_pct_at_map']}%."
        )
        chunks.append({"source": "pricing", "type": "pricing", "title": it["sku"], "sku": it["sku"], "text": text})

    comp = json.loads((DATA_DIR / "competitors.json").read_text(encoding="utf-8"))
    for c in comp["competitors"]:
        text = (
            f"Competitor {c['brand']} — flagship {c['flagship']} (ASIN {c['asin']}) at ${c['price_usd']}, "
            f"Amazon BSR #{c['amazon_bsr']}, rating {c['rating']} ({c['reviews']} reviews). "
            f"Positioning: {c['positioning']}. Recent move: {c['recent_move']}"
        )
        chunks.append({"source": "competitors", "type": "competitor", "title": c["brand"], "text": text})

    for fname in ("policies.md", "sops.md", "brand.md"):
        raw = (DATA_DIR / fname).read_text(encoding="utf-8")
        chunks.extend(_md_sections(raw, source=fname.replace(".md", "")))

    return chunks


def run() -> int:
    embedder = get_embedder()
    chunks = build_chunks()
    for i, ch in enumerate(chunks):
        ch["id"] = i

    print(f"[ingest] {len(chunks)} chunks | embeddings = {embedder.label}")
    vectors = embedder.embed_documents([c["text"] for c in chunks])

    client = vs.get_client()
    vs.ensure_collection(client, dim=embedder.dim)
    vs.upsert(
        client,
        ids=[c["id"] for c in chunks],
        vectors=vectors,
        payloads=[{k: c[k] for k in c if k != "id"} for c in chunks],
    )

    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest] indexed into Qdrant collection '{vs.COLLECTION}' and saved {CHUNKS_PATH.name}")
    return len(chunks)


if __name__ == "__main__":
    run()
