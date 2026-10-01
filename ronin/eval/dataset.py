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
