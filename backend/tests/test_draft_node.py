import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm
from nodes.draft import draft_node
from patent_analyzer.adjudicate import adjudicate
from patent_analyzer.draft import assemble

DOC = ("[S1.P1] Each drone broadcasts a 64-dimensional state embedding vector via ultra-wideband pulses at 10 Hz. "
       "[S1.P2] Each drone derives a local phase from the interference pattern of received pulses. "
       "[S1.P3] The embedding is trained with a self-supervised contrastive loss so that nearby drones converge. "
       "[S1.P4] In one variant the pulses are emitted only when the battery level exceeds 20 percent. "
       "[S1.P5] A watchdog timer resets the phase estimator after 500 ms without pulses.")
E = ["broadcasting, by each drone, a state embedding vector via ultra-wideband pulses",
     "deriving, by each drone, a local phase from an interference pattern of received pulses",
     "wherein the embedding vector is trained with a contrastive loss"]
CL = [{"id": f"inv1.e{i + 1}", "criterion": e, "weight": 1 / 3} for i, e in enumerate(E)]
EXT = {"candidate_inventions": [
    {"id": "inv1", "level": "core", "primary_form": "method", "dependent_hints": ["pulses emitted only when battery exceeds 20 percent"],
     "elements": [{"id": "inv1.e0", "text": "A method of coordinating a swarm of drones", "kind": "structure", "evidence_quote": "Each drone broadcasts",
                   "evidence_loc": {"char": [8, 30]}},
                  {"id": "inv1.e1", "text": E[0], "kind": "step", "evidence_quote": "broadcasts a 64-dimensional state embedding vector via ultra-wideband pulses at 10 Hz",
                   "evidence_loc": {"char": [19, 100]}},
                  {"id": "inv1.e2", "text": E[1], "kind": "step", "evidence_quote": "derives a local phase from the interference pattern of received pulses",
                   "evidence_loc": {"char": [120, 190]}},
                  {"id": "inv1.e3", "text": E[2], "kind": "condition", "evidence_quote": "trained with a self-supervised contrastive loss",
                   "evidence_loc": {"char": [230, 280]}}]},
    {"id": "inv2", "level": "component", "elements": [
        {"id": "inv2.e0", "text": "A phase estimator"},
        {"id": "inv2.e1", "text": "a watchdog timer resetting the phase estimator after 500 ms without pulses", "kind": "structure",
         "evidence_quote": "A watchdog timer resets the phase estimator after 500 ms without pulses", "evidence_loc": {"char": [300, 370]}}]}]}


def _doc(pub, covered):
    cr = {e: ({"score": 2, "verified_quotes": ["verbatim"]} if e in covered else {"score": 0, "verified_quotes": []}) for e in E}
    return {"pub_num": pub, "title": f"Patent {pub}", "checklist_results": cr}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setenv("DRAFT_RECHECK", "0")
    monkeypatch.setattr(assemble, "embed_similarity", assemble.difflib_similarity)
    calls = {"draft": 0, "eval": 0, "reword": 0, "advisory": 0}

    async def fake_draft(primary_form, primary, mirror, pool, refine_targets, coverage_lines, document_text):
        calls["draft"] += 1
        return {"primary": {l["lid"]: l["text"].replace("by each drone", "by each of the drones") for l in primary},
                "mirror": {l["lid"]: l["text"] for l in mirror},
                "pool": {"hint0": {"text": "wherein the pulses are emitted only when a battery level exceeds 20 percent",
                                   "evidence_quote": "the pulses are emitted only when the battery level exceeds 20 percent"}},
                "refinements": [{"element_id": "inv1.e1", "text": "wherein the pulses are emitted at 10 Hz", "evidence_quote": "ultra-wideband pulses at 10 Hz"},
                                {"element_id": "inv1.e2", "text": "wherein the phase converges within 200 ms", "evidence_quote": "converges within 200 ms"}]}

    async def fake_eval(summary, checklist, text, title, ptype, persona=None, doc_mode="abstract"):
        calls["eval"] += 1
        cr = {}
        for c in checklist:
            crit = c["criterion"]
            # US1 discloses the watchdog; nobody discloses the battery / 10 Hz limitations
            if "watchdog" in crit and "US1" in title:
                cr[crit] = {"score": 2, "evidence_quotes": ["watchdog timer resets"], "match": True}
            else:
                cr[crit] = {"score": 0, "evidence_quotes": [], "match": False}
        return {"checklist_results": cr, "title": title}

    async def fake_reword(flagged):
        calls["reword"] += 1
        return {f["lid"]: f["text"].replace("the interference pattern", "an interference pattern") for f in flagged}

    async def fake_advisory(claims, description, thinking_budget=2048):
        calls["advisory"] += 1
        return {c["no"]: {"likelihood": "unlikely", "p_indefinite": 0.2, "reasons": []} for c in claims}

    monkeypatch.setattr(llm, "draft_claims", fake_draft)
    monkeypatch.setattr(llm, "evaluate_single_document_text", fake_eval)
    monkeypatch.setattr(llm, "reword_limitations", fake_reword)
    monkeypatch.setattr(llm, "definiteness_advisory", fake_advisory)
    return calls


def _state(docs):
    adj = adjudicate(CL, docs, single_partial_103=0.7)
    return {"job_id": "t", "extraction": EXT, "checklist": CL, "scoring_report": docs, "adjudication": adj,
            "ranked_candidates": [{"pub_num": d["pub_num"], "title": d["title"], "abstract": "watchdog timer resets the phase estimator " * 6} for d in docs],
            "summary": "drone swarm", "document_text": DOC}


def test_102_draft_narrows_with_a_grounded_hint_and_every_limitation_has_basis(patched):
    out = asyncio.run(draft_node(_state([_doc("US1", E), _doc("US2", E[:1])])))
    d = out["draft_claims"]
    assert d["strategy"] == "narrowed" and d["basis_adjudication"]["label"] == "102" and d["candidate_id"] == "inv1"
    assert out["adjudication"]["claim_chart"]["docs"][0]["key"] == "US1"
    c1 = d["claims"][0]
    assert c1["no"] == 1 and c1["form"] == "method" and c1["preamble"].startswith("A method of coordinating a swarm of drones")
    assert len(c1["limitations"]) == 4                                   # 3 elements + the added limitation
    added = c1["limitations"][-1]
    assert added["origin"] == "dependent_hint" and added["distinguishing"] and added["coverage"]["covered_by"] == []
    assert added["basis"][0]["evidence_quote"].startswith("the pulses are emitted only when") and added["basis"][0]["evidence_loc"]["char"]
    for c in d["claims"]:
        for l in c["limitations"]:
            assert l["basis"] and l["basis"][0]["element_id"] and l["basis"][0]["evidence_quote"]
    assert c1["limitations"][0]["coverage"]["covered_by"] == ["US1", "US2"] and c1["limitations"][2]["coverage"]["covered_by"] == ["US1"]
    assert c1["limitations"][0]["text"] == E[0].replace("by each drone", "by each of the drones")     # accepted rewording
    # dependents: the 10 Hz refinement (undisclosed) then the watchdog element (disclosed by US1); the 200 ms refinement did not snap
    deps = [c for c in d["claims"] if c["depends_on"] == 1]
    assert [c["limitations"][0]["origin"] for c in deps] == ["refinement", "component_element"]
    assert deps[1]["limitations"][0]["coverage"]["covered_by"] == ["US1"]
    assert any(p["dropped"] == "unsupported" for p in d["avoidance"]["pool"] if p["pid"] == "ref1")
    mirror = next(c for c in d["claims"] if c["form"] == "system" and c["depends_on"] is None)
    assert mirror["no"] == 4 and len(mirror["limitations"]) == 4 and mirror["preamble"].startswith("A system for coordinating")
    assert [c["depends_on"] for c in d["claims"] if c["no"] > 4] == [4, 4]
    assert [c["no"] for c in d["claims"]] == [1, 2, 3, 4, 5, 6]
    assert d["definiteness"]["open_flags"] == [] and d["definiteness"]["llm_advisory"]["1"]["p_indefinite"] == 0.2
    assert patched == {"draft": 1, "eval": 2, "reword": 0, "advisory": 1} and d["llm_calls"] == 4
    assert d["recheck"]["skipped"] is True
    assert any(e["kind"] == "draft_plan" for e in out["events"]) and out["phase_results"]["phase4b"]["data"]["strategy"] == "narrowed"


def test_unresolved_when_every_pool_limitation_is_disclosed(patched, monkeypatch):
    async def all_disclosed(summary, checklist, text, title, ptype, persona=None, doc_mode="abstract"):
        return {"checklist_results": {c["criterion"]: {"score": 2, "evidence_quotes": ["watchdog timer resets"]} for c in checklist}}
    monkeypatch.setattr(llm, "evaluate_single_document_text", all_disclosed)
    out = asyncio.run(draft_node(_state([_doc("US1", E)])))
    d = out["draft_claims"]
    assert d["strategy"] == "unresolved" and len(d["claims"][0]["limitations"]) == 3
    assert "needs input from the inventor" in d["avoidance"]["reason"]


def test_no_prior_art_and_no_elements(patched):
    out = asyncio.run(draft_node({**_state([]), "scoring_report": [], "adjudication": {}}))
    d = out["draft_claims"]
    assert d["strategy"] == "no_prior_art" and d["claims"][0]["limitations"][0]["coverage"]["status"] == "unknown"
    assert patched["eval"] == 0
    out = asyncio.run(draft_node({"extraction": {"candidate_inventions": []}}))
    assert out["draft_claims"]["strategy"] == "no_elements"


def test_reword_round_runs_on_open_flags(patched):
    ext = {"candidate_inventions": [{**EXT["candidate_inventions"][0], "dependent_hints": [],
                                     "elements": EXT["candidate_inventions"][0]["elements"][:2]
                                     + [{"id": "inv1.e2", "text": "a module for estimating the phase substantially in real time", "kind": "structure",
                                         "evidence_quote": "derives a local phase from the interference pattern of received pulses", "evidence_loc": {"char": [120, 190]}}]}]}
    st = {**_state([_doc("US1", E[:1])]), "extraction": ext, "checklist": CL[:2]}
    out = asyncio.run(draft_node(st))
    d = out["draft_claims"]
    cats = {f["category"] for f in d["definiteness"]["flags"]}
    assert {"functional_claiming", "relative_term"} <= cats
    assert patched["reword"] >= 1 and d["definiteness"]["passes"] >= 2
