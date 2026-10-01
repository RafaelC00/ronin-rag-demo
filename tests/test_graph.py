"""Graph behaviour tests. The LLM and the retriever are faked, so these run offline and cost nothing."""
from __future__ import annotations

import pytest
from langgraph.types import Command

from ronin.agents import graph as G
from ronin.agents import knowledge as K
from ronin.agents import report as R
from ronin.config import get_settings
from ronin.retrieval import Passage


def P(i: int, text: str = "") -> Passage:
    return Passage(id=i, text=text or f"passage {i}", source="policies", title=f"t{i}")


class FakeLLM:
    """Scripted LLM. `grades` / `rewrites` are consumed in order; the last item repeats."""

    def __init__(self, grades=(True,), rewrites=("q2", "q3", "q4")):
        self.grades, self.rewrites = list(grades), list(rewrites)
        self.calls = {"grade": 0, "rewrite": 0, "answer": 0}

    @staticmethod
    def _next(items, n):
        return items[min(n, len(items) - 1)]

    def json(self, system, user, **kw):
        if "grade whether" in system:
            ok = self._next(self.grades, self.calls["grade"])
            self.calls["grade"] += 1
            if isinstance(ok, Exception):
                raise ok
            return {"sufficient": ok, "missing": "" if ok else "the number", "reason": "x"}
        if "rewrite a search query" in system:
            q = self._next(self.rewrites, self.calls["rewrite"])
            self.calls["rewrite"] += 1
            return {"query": q}
        if "Route a user request" in system:
            return {"route": "knowledge"}
        return {"headline": "h", "action": "a"}

    def complete(self, system, user, **kw):
        self.calls["answer"] += 1
        return "the answer"


@pytest.fixture
def env(monkeypatch):
    """Patch LLM + retriever everywhere they are imported; returns (llm, retrieve_log)."""
    log: list[str] = []

    def install(llm):
        for mod in (K, G, R):
            monkeypatch.setattr(mod, "get_llm", lambda: llm)
        monkeypatch.setattr(K, "retrieve", lambda q, top_k=4: (log.append(q), [P(len(log), f"for {q}")])[1])
        monkeypatch.setattr(R, "post_to_slack", lambda blocks: False)

    monkeypatch.setattr(get_settings(), "ronin_confirm_report", "auto")
    monkeypatch.setattr(get_settings(), "slack_webhook_url", "")
    return install, log


def test_weak_context_loops_back_then_answers(env):
    install, log = env
    llm = FakeLLM(grades=(False, True))
    install(llm)
    out = K.build_knowledge_graph().invoke({"question": "q1"})
    assert log == ["q1", "q2"]                      # the rewrite really re-queried
    assert out["attempts"] == 2 and out["stop_reason"] == "sufficient"
    assert out["queries"] == ["q1", "q2"]
    assert len(out["passages"]) == 2                # earlier hit kept alongside the new one
    assert llm.calls == {"grade": 2, "rewrite": 1, "answer": 1}


def test_sufficient_context_does_not_loop(env):
    install, log = env
    install(FakeLLM(grades=(True,)))
    out = K.build_knowledge_graph().invoke({"question": "q1"})
    assert log == ["q1"] and out["attempts"] == 1 and out["stop_reason"] == "sufficient"


def test_step_budget_bounds_the_cycle(env):
    install, log = env
    llm = FakeLLM(grades=(False,))                  # grader never satisfied
    install(llm)
    out = K.build_knowledge_graph().invoke({"question": "q1", "budget": 3})
    assert len(log) == 3 and out["attempts"] == 3
    assert out["stop_reason"] == "budget"
    assert llm.calls["answer"] == 1                 # still answers (and the generator may refuse)


def test_default_budget_comes_from_settings(env, monkeypatch):
    install, log = env
    install(FakeLLM(grades=(False,)))
    monkeypatch.setattr(get_settings(), "ronin_max_retrievals", 2)
    out = K.build_knowledge_graph().invoke({"question": "q1"})
    assert len(log) == 2 and out["budget"] == 2


def test_loop_stops_when_rewriter_repeats_itself(env):
    install, log = env
    install(FakeLLM(grades=(False,), rewrites=("q1",)))   # rewrite == the original query
    out = K.build_knowledge_graph().invoke({"question": "q1", "budget": 5})
    assert log == ["q1"] and out["stop_reason"] == "no_new_query"


def test_grader_failure_falls_through_to_answer(env):
    install, log = env
    install(FakeLLM(grades=(RuntimeError("boom"),)))
    out = K.build_knowledge_graph().invoke({"question": "q1"})
    assert log == ["q1"] and out["stop_reason"] == "grader_error" and out["answer"]


def test_supervisor_routes_and_knowledge_runs_through_the_cycle(env):
    install, log = env
    install(FakeLLM(grades=(False, True)))
    g = G.build_graph()
    out = g.invoke({"question": "what is our return window"})
    assert out["route"] == "knowledge" and out["attempts"] == 2
    assert out["steps"][0] == "supervisor → knowledge"
    assert any(s.startswith("rewrite") for s in out["steps"])


def test_report_gate_pauses_and_resumes_across_a_restart(env, monkeypatch, tmp_path):
    install, _ = env
    install(FakeLLM())
    monkeypatch.setattr(get_settings(), "ronin_confirm_report", "always")
    db = tmp_path / "ckpt.sqlite"
    cfg = {"configurable": {"thread_id": "t-1"}}

    first = G.build_graph(G.make_checkpointer(db))
    out = first.invoke({"question": "run the daily brief"}, cfg)
    assert out["__interrupt__"][0].value["action"] == "daily_brief"
    assert "report_text" not in out                  # nothing ran past the gate

    # "restart": a brand-new graph and a brand-new SQLite connection on the same file
    second = G.build_graph(G.make_checkpointer(db))
    snap = second.get_state(cfg)
    assert snap.next == ("confirm_report",)
    done = second.invoke(Command(resume=True), cfg)
    assert done["approved"] is True and done["report_text"] and "__interrupt__" not in done


def test_declining_the_gate_skips_the_report(env, monkeypatch, tmp_path):
    install, _ = env
    llm = FakeLLM()
    install(llm)
    monkeypatch.setattr(get_settings(), "ronin_confirm_report", "always")
    g = G.build_graph(G.make_checkpointer(tmp_path / "c.sqlite"))
    cfg = {"configurable": {"thread_id": "t-2"}}
    g.invoke({"question": "daily brief"}, cfg)
    out = g.invoke(Command(resume=False), cfg)
    assert out["approved"] is False and "cancelled" in out["answer"]
    assert "report_text" not in out and llm.calls["answer"] == 0


def test_gate_is_skipped_when_nothing_goes_outward(env, tmp_path):
    install, _ = env
    install(FakeLLM())
    g = G.build_graph(G.make_checkpointer(tmp_path / "c.sqlite"))   # confirm mode = auto, no webhook
    out = g.invoke({"question": "daily brief"}, {"configurable": {"thread_id": "t-3"}})
    assert "__interrupt__" not in out and out["report_text"]


def test_reused_thread_starts_each_question_clean(env, tmp_path):
    install, log = env
    install(FakeLLM(grades=(False, True, True)))
    g = G.build_graph(G.make_checkpointer(tmp_path / "c.sqlite"))
    cfg = {"configurable": {"thread_id": "t-4"}}
    a = g.invoke({"question": "first"}, cfg)
    b = g.invoke({"question": "second"}, cfg)
    assert a["attempts"] == 2 and b["attempts"] == 1
    assert b["queries"] == ["second"]
