from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    route: str                 # knowledge | intel | report

    # --- corrective-RAG loop ---
    query: str                 # the query currently being retrieved with (starts as the question)
    queries: list              # every query tried, in order
    attempts: int              # retrievals performed so far
    budget: int                # max retrievals allowed (the step budget)
    grade: dict                # latest grader verdict: {"sufficient", "missing", "reason"}
    stop_reason: str           # why the loop ended: sufficient | budget | no_new_query | grader_error

    passages: list             # list[dict] for serialization
    answer: str
    citations: list            # list[str]
    intel: dict
    report_blocks: list
    report_text: str
    approved: bool             # human decision at the report confirmation gate
    # nodes return only the steps they add; the reducer appends them
    steps: Annotated[list, operator.add]
