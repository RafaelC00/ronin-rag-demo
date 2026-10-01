"""Golden eval set: questions + the source(s) that should ground the answer."""

EVAL_SET = [
    {"question": "What's the MAP price and gross margin on the flagship Shogun competition gi?",
     "expected_sources": ["catalog", "pricing"],
     "must_include": ["129", "55"]},
    {"question": "How deep can we discount below MAP during a sale, and who signs off beyond that?",
     "expected_sources": ["policies"],
     "must_include": ["15%", "VP Sales"]},
    {"question": "What's our return policy on a washed gi?",
     "expected_sources": ["policies"],
     "must_include": ["unwashed"]},
    {"question": "What should we do when the flagship BSR drops several positions?",
     "expected_sources": ["sops"],
     "must_include": ["coupon", "MAP"]},
    {"question": "Which competitor just undercut our gi price and by how much?",
     "expected_sources": ["competitors"],
     "must_include": ["Osoto", "119"]},
    {"question": "What is the reorder floor for the Shogun gi and why?",
     "expected_sources": ["sops"],
     "must_include": ["800"]},
]

# ---------------------------------------------------------------------------
# Extension (written before any before/after run, and not edited since):
# paraphrased and multi-hop questions, plus two questions the knowledge base
# cannot answer. For those, `expected_sources` and `must_include` are empty and
# the right behaviour is to say so; only the LLM-judged metrics apply.
# ---------------------------------------------------------------------------
EVAL_SET += [
    {"question": "Can we run 20% off over the Black Friday weekend?",
     "expected_sources": ["policies"],
     "must_include": ["15%", "VP Sales"]},
    {"question": "How long does a customer have to send a gi back, and is Amazon different?",
     "expected_sources": ["policies"],
     "must_include": ["30"]},
    {"question": "What happens to a reseller who keeps advertising under MAP?",
     "expected_sources": ["policies"],
     "must_include": ["warning", "suspension"]},
    {"question": "How many weeks of stock do we keep on belts?",
     "expected_sources": ["sops"],
     "must_include": ["4"]},
    {"question": "Which product line earns us the best margin at MAP?",
     "expected_sources": ["pricing", "catalog"],
     "must_include": ["71"]},
    {"question": "The flagship rank slipped 5 spots and a rival cut price: what steps do we follow and what is the biggest coupon we can offer?",
     "expected_sources": ["sops"],
     "must_include": ["coupon", "15%"]},
    {"question": "How much did we spend on TikTok Shop ads last quarter?",
     "expected_sources": [],
     "must_include": []},
    {"question": "Who is the head of the customer support team and what is their email?",
     "expected_sources": [],
     "must_include": []},
]
