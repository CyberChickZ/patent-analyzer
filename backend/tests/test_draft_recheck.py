import asyncio
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm
from patent_analyzer.agentic import elements as agentic_elements
from patent_analyzer.agentic import loop as agentic_loop
from patent_analyzer.agentic import prune as agentic_prune
from patent_analyzer.draft import recheck as R
from patent_analyzer.recall.pool import Candidate


def _lim(lid, text, origin, covered_by=()):
    return {"lid": lid, "text": text, "origin": origin, "basis": [{"element_id": "x", "evidence_quote": "q"}],
            "coverage": {"covered_by": list(covered_by), "checked_against": ["US1"], "verified": True}, "flags": []}


DRAFT = {"strategy": "narrowed", "claims": [
    {"no": 1, "form": "method", "depends_on": None, "preamble": "A method, comprising:",
     "limitations": [_lim("c1.l1", "broadcasting a vector", "element", ["US1"]), _lim("c1.l2", "wherein the pulses are emitted at 10 Hz", "refinement")]},
    {"no": 2, "form": "method", "depends_on": 1, "preamble": "The method of claim 1, wherein",
     "limitations": [_lim("c2.l1", "the battery level exceeds 20 percent", "dependent_hint")]},
    {"no": 3, "form": "method", "depends_on": 1, "preamble": "The method of claim 1, further comprising",
     "limitations": [_lim("c3.l1", "a watchdog timer resetting the estimator", "component_element", ["US1"])]},
    {"no": 4, "form": "system", "depends_on": None, "preamble": "A system, comprising:",
     "limitations": [_lim("c4.l1", "a transmitter", "element", ["US1"]), _lim("c4.l2", "wherein the pulses are emitted at 10 Hz", "refinement")]},
    {"no": 5, "form": "system", "depends_on": 4, "preamble": "The system of claim 4, wherein",
     "limitations": [_lim("c5.l1", "the battery level exceeds 20 percent", "dependent_hint")]}]}
STATE = {"summary": "drones", "date_cutoff": "20240101",
         "extraction": {"candidate_inventions": [{"id": "inv1", "level": "core", "cpc_pred": ["G05D1"], "elements": [
             {"id": "inv1.e0", "text": "A method of coordinating drones", "facets": {"thing": ["drone swarm", "uwb"]}}]}]},
         "search_stats": {"pool": [{"pub_num": "US1"}], "serpapi_left": 0}, "ranked_candidates": [{"pub_num": "US1"}]}


@pytest.fixture
def patched(monkeypatch):
    seen = {"queries": [], "evaluated": [], "facets": 0}

    async def fake_attach(els, summary):
        seen["facets"] += 1
        for e in els:
            e["facets"] = {"thing": [e["text"].split()[-1], "pulse"], "place": [], "apparatus": []}
        return els

    async def fake_search(query, before, budget, num=20, scholar=False):
        seen["queries"].append((query, before, num))
        budget.gp_calls += 1
        return [Candidate(title="Old", pub_num="US1", match_type="Patent"),
                Candidate(title="New A", pub_num="US9A", match_type="Patent", abstract="pulses at 10 Hz"),
                Candidate(title="New B", pub_num="US9B", match_type="Patent", abstract="battery")], 3, "google_patents"

    def fake_stage1(els, docs, topk=5, **kw):
        for i, d in enumerate(docs):
            d["prune_cos"] = 0.9 - 0.1 * i
        return list(range(len(docs))), {}

    async def fake_fetch(doc):
        return f"FULL TEXT of {doc['pub_num']}: pulses are emitted at 10 Hz by the transmitter " * 4, "google_patents_page"

    async def fake_eval(summary, checklist, text, title, ptype, persona=None, doc_mode="abstract"):
        seen["evaluated"].append((title, [c["id"] for c in checklist], doc_mode))
        cr = {}
        for c in checklist:
            hit = "US9A" in text and "10 Hz" in c["criterion"]
            cr[c["criterion"]] = {"score": 2 if hit else 0, "evidence_quotes": ["pulses are emitted at 10 Hz"] if hit else []}
        return {"checklist_results": cr}

    monkeypatch.setattr(agentic_elements, "attach_facets", fake_attach)
    monkeypatch.setattr(agentic_loop, "_search", fake_search)
    monkeypatch.setattr(agentic_prune, "stage1_embed", fake_stage1)
    monkeypatch.setattr(R, "_fetch_text", fake_fetch)
    monkeypatch.setattr(llm, "evaluate_single_document_text", fake_eval)
    return seen


def test_new_limitations_are_the_independent_and_first_three_dependents():
    got = [(t["claim"]["no"], t["lim"]["lid"]) for t in R.new_limitations(DRAFT)]
    assert got == [(1, "c1.l2"), (2, "c2.l1"), (3, "c3.l1")]
    assert R.new_limitations({"claims": []}) == []


def test_recheck_dedupes_evaluates_new_docs_and_swaps_a_disclosed_independent_limitation(patched):
    draft = copy.deepcopy(DRAFT)
    out = asyncio.run(R.recheck(draft, STATE))
    assert patched["facets"] == 1 and 1 <= len(out["queries"]) <= R.MAX_QUERIES
    assert all(q[1] == "priority:20240101" and q[2] == 50 for q in patched["queries"])
    assert out["queries"][0]["new"] == 2 and "US1" not in out["queries"][0]["new_pubs"]      # already in the pool
    assert [d["pub_num"] for d in out["new_docs"]] == ["US9A", "US9B"] and out["evaluated"] == 2
    assert all(e[2] == "full_text" and e[1] == ["c1.l2", "c2.l1", "c3.l1"] for e in patched["evaluated"])
    assert out["limitation_coverage"] == {"c1.l2": ["US9A"], "c2.l1": [], "c3.l1": []}
    assert out["llm_calls"] == 3 and out["gp_calls"] == len(out["queries"]) and out["serp_calls"] == 0
    # the 10 Hz limitation added to claim 1 was disclosed by US9A -> the battery limitation (undisclosed by US1 and the new docs) takes its place
    assert out["swapped"]["out"] == "wherein the pulses are emitted at 10 Hz" and out["swapped"]["in"] == "the battery level exceeds 20 percent"
    assert draft["claims"][0]["limitations"][1]["text"] == "the battery level exceeds 20 percent" and draft["claims"][0]["limitations"][1]["distinguishing"]
    assert draft["claims"][1]["limitations"][0]["text"] == "wherein the pulses are emitted at 10 Hz"
    assert draft["claims"][3]["limitations"][1]["text"] == "the battery level exceeds 20 percent"        # mirror follows
    assert draft["claims"][1]["limitations"][0]["coverage"]["recheck"]["covered_by_new"] == ["US9A"]
    assert "strategy" not in out


def test_recheck_unresolved_when_no_dependent_survives(patched, monkeypatch):
    async def all_hit(summary, checklist, text, title, ptype, persona=None, doc_mode="abstract"):
        return {"checklist_results": {c["criterion"]: {"score": 2, "evidence_quotes": ["pulses are emitted at 10 Hz"]} for c in checklist}}
    monkeypatch.setattr(llm, "evaluate_single_document_text", all_hit)
    draft = copy.deepcopy(DRAFT)
    out = asyncio.run(R.recheck(draft, STATE))
    assert out["strategy"] == "unresolved" and "not narrowed by a verified limitation" in out["reason"]
    assert draft["claims"][0]["limitations"][1]["text"] == "wherein the pulses are emitted at 10 Hz"


def test_recheck_with_nothing_new_or_no_targets(patched, monkeypatch):
    async def only_known(query, before, budget, num=20, scholar=False):
        budget.gp_calls += 1
        return [Candidate(title="Old", pub_num="US1", match_type="Patent")], 1, "google_patents"
    monkeypatch.setattr(agentic_loop, "_search", only_known)
    draft = copy.deepcopy(DRAFT)
    out = asyncio.run(R.recheck(draft, STATE))
    assert out["new_docs"] == [] and out["evaluated"] == 0 and out["reason"].startswith("no new documents")
    assert draft["claims"][0]["limitations"][1]["coverage"]["recheck"] == {"queried": True, "new_docs": 0, "covered_by_new": []}
    out = asyncio.run(R.recheck({"claims": [{"no": 1, "depends_on": None, "limitations": [_lim("c1.l1", "x", "element")]}]}, STATE))
    assert out["skipped"] and out["reason"] == "no new limitations to re-check"


def test_serp_budget_reads_search_stats_or_a_quarter_of_the_job_cap(monkeypatch):
    assert R._serp_budget({"search_stats": {"serpapi_left": 2}}) == 2
    assert R._serp_budget({"search_stats": {"serpapi_left": 9}}) == R.MAX_QUERIES
    monkeypatch.setenv("SERPAPI_MAX_CALLS_PER_JOB", "3")
    assert R._serp_budget({}) == 0
    monkeypatch.setenv("SERPAPI_MAX_CALLS_PER_JOB", "8")
    assert R._serp_budget({}) == 2
