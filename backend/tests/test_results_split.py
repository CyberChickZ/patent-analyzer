"""results.json keeps what is read; funnel.json keeps everything.

Measured on real jobs before the split: results.json was 20.4 MB (job 075b99c1)
and 12.1 MB (c7b9e7bf), of which search_stats was 12.0 / 7.3 MB — loop_rounds
6.6 MB with cited_by_seed alone at 5.4 MB over 5,073 seeds, and funnel_docs
4.5 MB over 10,097 documents. The Express proxy does `await res.json()` and
re-serialises, so those bytes are paid for twice.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from patent_analyzer.funnel import slim_search_stats as _slim_search_stats
from patent_analyzer.report_sections import loop_html

STATS = {
    "total_patents": 12, "total_papers": 3, "total_unique": 15, "active_channels": 4, "downloaded": 2,
    "loop_mode": "wide", "loop_elements": [{"id": "e1", "text": "an element"}],
    "channel_health": [{"channel": "arxiv", "status": "ok", "n": 5}],
    "serpapi_quota": [{"key": "abcd", "used": 3, "cap": 250}],
    "prune": {"pool": 15, "stage1_out": 9, "stage2_worth": 4, "stage2_calls": 2},
    "pruned": [{"pub_num": "US1", "sources": ["x"], "match_type": "Patent", "elements": ["e1"]}],
    "pool": [{"pub_num": f"US{i}", "sources": ["s"], "match_type": "Patent"} for i in range(500)],
    "funnel_docs": ([{"pub_num": f"US{i}", "title": "t" * 100, "cos": 0.4, "reason": "r" * 200,
                      "elements": ["e1"], "rank": None} for i in range(500)]
                    + [{"pub_num": "US1", "title": "kept", "elements": ["e1"], "rank": 1, "reason": "r" * 200}]),
    "loop_rounds": [{
        "round": 1, "n_queries": 1, "gp_calls": 1, "serpapi_calls": 0, "seeds": 3, "expanded": 9,
        "pool_size": 15, "covered": ["e1"], "uncovered": [], "cpc_hint": "H04N",
        "queries": [{"n": 1, "kind": "react", "query": "q", "total": 7, "hits": 2, "new": 2,
                     "pubs": ["US1", "US2"], "facets_used": {}, "elements": ["e1"]}],
        "cited_by_seed": {f"US{i}": [f"US{j}" for j in range(20)] for i in range(300)},
        "bridge_by_paper": {"W1": ["US1"]}, "seed_pubs": [f"US{i}" for i in range(400)],
        "bridge_pubs": ["US1"], "pool_pubs": [f"US{i}" for i in range(500)],
        "expanded_pubs": ["US7"], "similar_by_seed": {"US1": ["US8"]}, "similar_pubs": ["US8"],
        "neigh_oa_ids": ["W1"], "lens_pubs": ["US9"],
        "lens": {"bridge_patents": 2, "search_calls": 1, "searches": [{"terms": ["x"], "pubs": ["US9"]}]},
    }],
}


def _size(o):
    return len(json.dumps(o, ensure_ascii=False, default=str))


def test_the_split_is_where_the_bytes_are():
    slim = _slim_search_stats(STATS, "job1")
    assert _size(slim) < _size(STATS) / 10, "the split has to actually move the weight"
    r = slim["loop_rounds"][0]
    for k in ("cited_by_seed", "seed_pubs", "pool_pubs", "bridge_pubs", "expanded_pubs",
              "similar_by_seed", "similar_pubs", "neigh_oa_ids", "lens_pubs", "bridge_by_paper"):
        assert k not in r, k
    assert "pool" not in slim
    assert "searches" not in r["lens"] and r["lens"]["bridge_patents"] == 2   # counts stay


def test_counts_and_everything_the_report_reads_survive():
    slim = _slim_search_stats(STATS, "job1")
    r = slim["loop_rounds"][0]
    for k in ("round", "n_queries", "gp_calls", "serpapi_calls", "seeds", "expanded",
              "pool_size", "covered", "uncovered", "cpc_hint", "queries"):
        assert k in r, k
    assert r["queries"][0]["pubs"] == ["US1", "US2"]           # the per-query table needs these
    for k in ("channel_health", "serpapi_quota", "prune", "pruned", "loop_elements", "total_unique"):
        assert k in slim, k
    # the section renders byte-identically from the slim stats
    assert loop_html(slim) == loop_html(STATS)


def test_funnel_docs_keeps_only_what_is_looked_up():
    slim = _slim_search_stats(STATS, "job1")
    fd = slim["funnel_docs"]
    kept = {d["pub_num"] for d in fd}
    assert "US1" in kept and "US2" in kept          # referenced by a query
    assert "US400" not in kept                      # neither ranked nor referenced
    assert all(set(d) == {"pub_num", "rank", "elements"} for d in fd), "projection only"
    assert len(fd) < len(STATS["funnel_docs"]) / 10


def test_funnel_ref_says_where_the_rest_went():
    slim = _slim_search_stats(STATS, "jobX")
    ref = slim["funnel_ref"]
    assert ref["file"] == "funnel.json"
    assert ref["endpoint"] == "/api/jobs/jobX/funnel"
    assert ref["n_funnel_docs"] == len(STATS["funnel_docs"]) and ref["n_pool"] == 500


def test_empty_stats_do_not_explode():
    assert _slim_search_stats({}, "j")["funnel_docs"] == []
    assert _slim_search_stats({"loop_rounds": []}, "j")["loop_rounds"] == []


# ── the endpoints ──

def _client(monkeypatch, job, tmp_path):
    import app.main as m
    monkeypatch.setattr(m, "_save_job", lambda j: None)
    job["output_dir"] = str(tmp_path)
    m.jobs[job["id"]] = job
    return TestClient(m.app)


def test_funnel_endpoint_serves_the_file(monkeypatch, tmp_path):
    (tmp_path / "funnel.json").write_text(json.dumps({"job_id": "jf1", "search_stats": STATS}))
    c = _client(monkeypatch, {"id": "jf1", "status": "completed"}, tmp_path)
    r = c.get("/api/jobs/jf1/funnel")
    assert r.status_code == 200
    assert r.json()["search_stats"]["loop_rounds"][0]["cited_by_seed"]["US0"]   # the full detail is here


def test_funnel_endpoint_404s_cleanly(monkeypatch, tmp_path):
    c = _client(monkeypatch, {"id": "jf2", "status": "running"}, tmp_path)
    assert c.get("/api/jobs/jf2/funnel").status_code == 404
    assert c.get("/api/jobs/nope/funnel").status_code == 404


def test_usage_endpoint_reads_the_job_record(monkeypatch, tmp_path):
    job = {"id": "ju1", "status": "completed", "phase": "phase5",
           "phase_metrics": {"idca": {"seconds": 3.0, "llm_calls": 2, "cost_usd": 0.01}},
           "cost": {"totals": {"llm_calls": 2, "cost_usd": 0.01}, "bigquery_usd_per_tib": 6.25,
                    "prices_usd_per_mtok": {"gemini-2.5-pro": {"input": 1.25, "output": 10.0}},
                    "note": "Estimated from list prices; ... Not a bill."}}
    c = _client(monkeypatch, job, tmp_path)
    r = c.get("/api/jobs/ju1/usage").json()
    assert r["phases"]["idca"]["llm_calls"] == 2
    assert r["totals"]["cost_usd"] == 0.01
    assert r["prices_usd_per_mtok"]["gemini-2.5-pro"]["input"] == 1.25
    assert "not a bill" in r["note"].lower()


def test_usage_endpoint_falls_back_to_results_json(monkeypatch, tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(
        {"cost": {"phases": {"search": {"llm_calls": 9}}, "totals": {"llm_calls": 9}}}))
    c = _client(monkeypatch, {"id": "ju2", "status": "completed"}, tmp_path)
    r = c.get("/api/jobs/ju2/usage").json()
    assert r["phases"]["search"]["llm_calls"] == 9


def test_usage_endpoint_404s_when_nothing_was_recorded(monkeypatch, tmp_path):
    c = _client(monkeypatch, {"id": "ju3", "status": "queued"}, tmp_path)
    assert c.get("/api/jobs/ju3/usage").status_code == 404


# ── the job record (state.json) ──
#
# state.json is rewritten to local disk AND to GCS on every heartbeat, so it is
# the more expensive of the two records. Measured on job 075b99c1 before this:
# 18.6 MB — phases.phase3 12.0 MB (the same search_stats) plus one round_done
# event whose payload was 6.6 MB.

from patent_analyzer.funnel import EVENT_PAYLOAD_MAX, slim_event  # noqa: E402

BIG_EVENT = {"ts": "t", "phase": "phase3", "kind": "round_done", "message": "wide: 5 queries",
             "payload": {"round": 1, "seeds": 14109, "pool_size": 10097, "cpc_hint": "H04N",
                         "cited_by_seed": {f"US{i}": [f"US{j}" for j in range(20)] for i in range(300)},
                         "seed_pubs": [f"US{i}" for i in range(4000)]}}


def test_big_event_payloads_are_replaced_by_a_pointer():
    out = slim_event(BIG_EVENT)
    assert _size(out) < _size(BIG_EVENT) / 50
    assert set(out["payload_trimmed"]) == {"cited_by_seed", "seed_pubs"}
    # the counters a reader actually wants are untouched
    assert out["payload"]["seeds"] == 14109 and out["payload"]["pool_size"] == 10097
    assert out["payload"]["cpc_hint"] == "H04N" and out["message"] == BIG_EVENT["message"]
    # and the omission says so rather than looking like an empty result
    om = out["payload"]["cited_by_seed"]
    assert om["_omitted"] is True and om["n"] == 300 and "funnel.json" in om["where"]


def test_small_events_pass_through_untouched():
    e = {"ts": "t", "kind": "channel_done", "message": "arxiv: 5",
         "payload": {"channel": "arxiv", "n": 5, "errors": []}}
    assert slim_event(e) is e
    assert slim_event({"ts": "t", "kind": "info", "message": "x"}) == {"ts": "t", "kind": "info", "message": "x"}
    assert "payload_trimmed" not in slim_event(e)


def test_payload_limit_is_a_real_bound():
    e = {"kind": "k", "payload": {"a": ["x" * 10] * 5}}
    assert slim_event(e, limit=10)["payload"]["a"]["_omitted"] is True
    assert slim_event(e, limit=10_000)["payload"]["a"] == ["x" * 10] * 5
    assert EVENT_PAYLOAD_MAX >= 1024, "too small a cap would gut ordinary progress events"
