from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    route: str                 # knowledge | intel | report
    passages: list             # list[dict] for serialization
    answer: str
    citations: list            # list[str]
    intel: dict
    report_blocks: list
    report_text: str
    steps: list                # trace of nodes/decisions hit
