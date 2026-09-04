#!/usr/bin/env python3
"""Open-world retrieval harness (E4): paper in, examiner citations as gold.

Sample: Pap2Pat test pairs (paper full text <-> its own US application).
Gold: examiner citations (category contains 'SEA') of the application's
whole family, resolved to family_ids; filtered by per-query priority_date
cutoff (no forward art), self-family removed, applicant-only (IDS)
citations excluded by construction.

Stages:
  --stage gold     build + cache gold, print sanity columns (no LLM)
  --stage pipeline run IDCA -> SSR -> search_node per paper, score
                   family-level Recall@k, reach-vs-ranking, channel
                   unique contribution.
"""

import argparse
import asyncio
import json
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import os

PAP2PAT = Path(os.environ.get("PAP2PAT_DIR", "/tmp/pap2pat/Pap2Pat/data"))
RUN_DIR = Path(os.environ.get("E4_RUN_DIR", str(Path(__file__).parent.parent / "eval_data" / "runs" / "e4")))
GOLD_PATH = RUN_DIR / "gold.json"


def gcs_bundle_pull(uri: str):
    """Download bundle.tar.gz (gold.json + papers/<pair_id>/paper.json +
    metadata.json) from GCS into RUN_DIR / PAP2PAT for a Cloud Run Job."""
    import tarfile
    from google.cloud import storage
    bucket, _, key = uri[5:].partition("/")
    local = Path("/tmp/e4_bundle.tar.gz")
    storage.Client().bucket(bucket).blob(key).download_to_filename(local)
    with tarfile.open(local) as tf:
        tf.extractall("/tmp/e4_bundle")
    global PAP2PAT
    PAP2PAT = Path("/tmp/e4_bundle/papers")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    src = Path("/tmp/e4_bundle/gold.json")
    if src.exists() and not GOLD_PATH.exists():
        GOLD_PATH.write_text(src.read_text())


def gcs_bundle_push(uri: str):
    """Build the bundle from local Pap2Pat data + cached gold and upload."""
    import tarfile
    from google.cloud import storage
    gold = json.loads(GOLD_PATH.read_text())
    local = Path("/tmp/e4_bundle_out.tar.gz")
    with tarfile.open(local, "w:gz") as tf:
        tf.add(GOLD_PATH, arcname="gold.json")
        tf.add(PAP2PAT / "metadata.json", arcname="papers/metadata.json")
        for g in gold.values():
            tf.add(PAP2PAT / g["pair_id"] / "paper.json", arcname=f"papers/{g['pair_id']}/paper.json")
    bucket, _, key = uri[5:].partition("/")
    storage.Client().bucket(bucket).blob(key).upload_from_filename(local)
    print(f"bundle uploaded: {uri} ({local.stat().st_size // 1024} KB)")


def gcs_results_push(uri_prefix: str):
    from google.cloud import storage
    bucket, _, prefix = uri_prefix[5:].partition("/")
    b = storage.Client().bucket(bucket)
    n = 0
    for f in RUN_DIR.glob("*.json"):
        b.blob(f"{prefix.rstrip('/')}/{f.name}").upload_from_filename(f)
        n += 1
    print(f"uploaded {n} result files to {uri_prefix}")


def _canon(p: str) -> str:
    p = re.sub(r"[\s\-/,.]", "", (p or "").upper())
    m = re.match(r"^US(\d{10})([A-Z]\d?)?$", p)
    if m and m.group(1)[:2] in ("19", "20"):
        return f"US{m.group(1)[:4]}0{m.group(1)[4:]}{m.group(2) or ''}"
    return p


def sample_pairs(n: int, seed: int = 42) -> list[dict]:
    meta = json.loads((PAP2PAT / "metadata.json").read_text())
    test_ids = meta["splits"]["test"]
    rng = random.Random(seed)
    picked = sorted(rng.sample(test_ids, min(n, len(test_ids))))
    out = []
    for pid in picked:
        s = meta["samples"][pid]
        pub = s["patent"]["id"]
        if re.fullmatch(r"US\d{11}", pub):
            pub += "A1"  # Pap2Pat stores pre-grant publications without the kind code
        out.append({"pair_id": pid, "patent_pub": pub,
                    "application_date": s["patent"].get("application_date", ""),
                    "paper_title": s["paper"].get("title", "")})
    return out


def render_paper(pair_id: str) -> str:
    paper = json.loads((PAP2PAT / pair_id / "paper.json").read_text())
    parts = [f"Title: {paper.get('title', '')}", "", "Abstract", paper.get("abstract", ""), ""]

    def walk(secs, depth=1):
        for s in secs or []:
            parts.append("#" * depth + " " + (s.get("title") or ""))
            parts.extend(p for p in (s.get("paragraphs") or []) if p)
            walk(s.get("subsections"), depth + 1)
    walk(paper.get("sections"))
    return "\n".join(parts) + "\n"


async def build_gold(pairs: list[dict]) -> dict:
    from patent_analyzer.recall.bigquery_patents import fetch_by_pub_nums, fetch_citations, fetch_families

    pubs = [p["patent_pub"] for p in pairs]
    meta = await fetch_by_pub_nums(pubs, with_claims=False)
    fam_of = {k: v["family_id"] for k, v in meta.items()}
    families = await fetch_families(list({f for f in fam_of.values() if f}))

    # all family members' citations (the A1 alone often carries none)
    member_pubs = sorted({m["publication_number"] for ms in families.values() for m in ms})
    cits = await fetch_citations(member_pubs)

    gold = {}
    for p in pairs:
        key = _canon(p["patent_pub"])
        fam = fam_of.get(key, "")
        members = families.get(fam, [])
        prio = min((m["priority_date"] for m in members if m["priority_date"]), default=meta.get(key, {}).get("priority_date", ""))
        sea, app_only = set(), set()
        for m in members:
            for c in cits.get(m["publication_number"], {}).get("cits", []):
                if not c["cited"] or c["npl_text"]:
                    continue
                if "SEA" in c["category"]:
                    sea.add(c["cited"])
                elif "APP" in c["category"]:
                    app_only.add(c["cited"])
        gold[key] = {"pair_id": p["pair_id"], "family_id": fam, "priority_date": prio,
                     "n_members": len(members), "sea_cited": sorted(sea),
                     "app_only_cited": sorted(app_only - sea)}

    # resolve cited pubs -> family + priority (for cutoff / self-family / in-corpus)
    cited_all = sorted({c for g in gold.values() for c in g["sea_cited"]})
    cited_meta = await fetch_by_pub_nums(cited_all, with_claims=False) if cited_all else {}
    for g in gold.values():
        kept, dropped = [], {"self_family": 0, "forward": 0, "not_in_corpus": 0, "undated": 0}
        for c in g["sea_cited"]:
            cm = cited_meta.get(_canon(c))
            if cm is None:
                dropped["not_in_corpus"] += 1
                continue
            if cm["family_id"] and cm["family_id"] == g["family_id"]:
                dropped["self_family"] += 1
                continue
            if not cm["priority_date"] or not g["priority_date"]:
                dropped["undated"] += 1
            elif cm["priority_date"] >= g["priority_date"]:
                dropped["forward"] += 1
                continue
            kept.append({"pub": _canon(c), "family_id": cm["family_id"], "priority_date": cm["priority_date"],
                         "title": cm["title"][:80]})
        g["gold"] = kept
        g["gold_families"] = sorted({k["family_id"] for k in kept if k["family_id"]})
        g["dropped"] = dropped
    return gold


def print_sanity(gold: dict):
    n = len(gold)
    with_gold = sum(1 for g in gold.values() if g["gold"])
    tot = sum(len(g["gold"]) for g in gold.values())
    fams = sum(len(g["gold_families"]) for g in gold.values())
    drops = {k: sum(g["dropped"][k] for g in gold.values()) for k in ("self_family", "forward", "not_in_corpus", "undated")}
    sea_raw = sum(len(g["sea_cited"]) for g in gold.values())
    app_raw = sum(len(g["app_only_cited"]) for g in gold.values())
    print(f"queries={n}  with>=1 gold={with_gold}  gold docs={tot} (families={fams})  "
          f"raw SEA cites={sea_raw}  applicant-only cites (excluded)={app_raw}")
    print(f"dropped: {drops}")
    print("reachable upper bound (gold in corpus, after cutoff) = "
          f"{tot}/{sea_raw - drops['self_family'] - drops['forward']} ")
    print("random baseline R@100 ≈ 100 / 94.6M ≈ 0.000001 (family-level ~0.000001)")


async def run_pipeline_one(key: str, g: dict) -> dict:
    """IDCA -> Phase 2 (extraction subgraph) -> search_node; cached per query."""
    from graph.extraction_subgraph import build_extraction_subgraph
    from nodes.idca import idca_node
    from nodes.search import search_node

    tag = os.environ.get("E4_TAG", "")
    out_path = RUN_DIR / f"{key}_search{('_' + tag) if tag else ''}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(render_paper(g["pair_id"]))
        tmp = f.name
    p1 = await idca_node({"input_local_path": tmp})
    rec = {"key": key, "status_determination": p1.get("status_determination"), "input_mode": p1.get("input_mode"),
           "summary": p1.get("summary", ""), "delegation": {}, "ranked": [], "pool": [], "events": []}
    if p1.get("status_determination") == "Present":
        p2 = await build_extraction_subgraph().ainvoke({
            "summary": p1["summary"], "document_text": p1.get("document_text", ""), "input_local_path": tmp,
            "input_mode": p1.get("input_mode", "academic_paper"), "cpc_subclass": p1.get("cpc_subclass", "")})
        rec["delegation"] = p2.get("delegation", {})
        rec["extraction"] = p2.get("extraction")
        p3 = await search_node({
            "summary": p1["summary"], "source_title": p1.get("source_title", ""),
            "source_arxiv_id": p1.get("source_arxiv_id", ""), "source_doi": p1.get("source_doi", ""),
            "delegation": rec["delegation"], "checklist": p2.get("checklist", []),
            "extraction": rec["extraction"],
            "date_cutoff": g.get("priority_date") or None,
            "output_dir": str(RUN_DIR / "pdf" / key)})
        rec["loop_rounds"] = (p3.get("search_stats") or {}).get("loop_rounds", [])
        for k in ("move_rows", "good", "coverage", "stop", "uncovered", "n_strong", "read"):   # LOOP_MODE=moves (M1)
            rec[k] = (p3.get("search_stats") or {}).get(k)
        rec["serpapi_quota"] = (p3.get("search_stats") or {}).get("serpapi_quota", [])
        rec["ranked"] = [{"pub_num": d.get("pub_num", ""), "match_type": d.get("match_type", ""),
                          "sources": d.get("sources", []), "title": d.get("title", "")[:80]}
                         for d in p3.get("ranked_candidates", [])]
        rec["pool"] = (p3.get("search_stats") or {}).get("pool", [])
        rec["pruned"] = (p3.get("search_stats") or {}).get("pruned", [])
        rec["funnel_docs"] = (p3.get("search_stats") or {}).get("funnel_docs", [])
        rec["loop_elements"] = (p3.get("search_stats") or {}).get("loop_elements", [])
        rec["prune"] = (p3.get("search_stats") or {}).get("prune", {})
        rec["loop_mode"] = (p3.get("search_stats") or {}).get("loop_mode", "")
        rec["events"] = [e.get("message", "") for e in p3.get("events", [])
                         if e.get("kind") in ("channel_done", "channel_crashed", "channel_limited")]
        rec["channel_stats"] = [e.get("payload") for e in p3.get("events", []) if e.get("kind") == "channel_done"]
    from app import llm as _llm
    rec["llm_usage"] = {m: dict(u) for m, u in _llm.usage.items()}
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


async def score_pipeline(gold: dict, recs: dict) -> dict:
    """Family-level recall over the ranked list and the pool; channel attribution."""
    from patent_analyzer.recall.bigquery_patents import fetch_families

    # Family of a pool pub only matters when it is a gold family: expand the
    # gold families to their members (tens of rows) instead of looking up every
    # pool pub (thousands of rows, ~0.03 GiB each on the bucketed table).
    gold_fams = sorted({f for k in recs for f in gold.get(k, {}).get("gold_families", [])})
    members = await fetch_families(gold_fams) if gold_fams else {}
    fam_of = {_canon(m["publication_number"]): fid for fid, ms in members.items() for m in ms}
    for k in recs:
        for gd in gold.get(k, {}).get("gold", []):
            fam_of.setdefault(_canon(gd["pub"]), gd["family_id"])

    per_q, chan_unique, chan_hits = [], {}, {}
    for key, g in gold.items():
        gf = set(g["gold_families"])
        if not gf:
            continue
        r = recs.get(key)
        if not r:
            continue
        def fams(docs):
            return [fam_of.get(_canon(d["pub_num"]), "") for d in docs]
        ranked_f = fams(r["ranked"])
        pool_f = fams(r["pool"])
        pruned_f = fams(r.get("pruned") or [])
        hit_at = lambda lst, k: len(gf & {f for f in lst[:k] if f})
        pool_hit = gf & {f for f in pool_f if f}
        pruned_hit = gf & {f for f in pruned_f if f}
        rounds = r.get("loop_rounds") or []
        serp_calls = rounds[-1].get("serpapi_calls", 0) if rounds else 0
        gp_calls = rounds[-1].get("gp_calls", 0) if rounds else 0
        for d, f in zip(r["pool"], pool_f):
            if f in gf:
                for src in d.get("sources", []):
                    chan_hits[src] = chan_hits.get(src, 0) + 1
                if len(d.get("sources", [])) == 1:
                    chan_unique[d["sources"][0]] = chan_unique.get(d["sources"][0], 0) + 1
        per_q.append({"key": key, "n_gold_fam": len(gf), "pool": len(r["pool"]), "ranked": len(r["ranked"]),
                      "pool_hit": len(pool_hit), "r10": hit_at(ranked_f, 10), "r30": hit_at(ranked_f, 30),
                      "r100_pool": len(pool_hit),
                      "pruned": len(r.get("pruned") or []), "pruned_hit": len(pruned_hit) if r.get("pruned") else None,
                      "serpapi_calls": serp_calls, "gp_calls": gp_calls,
                      "prune_calls": (r.get("prune") or {}).get("stage2_calls", 0)})
    # reach by loop round (family-level, cumulative over the loop's own pool)
    by_round = {}
    for key, g in gold.items():
        gf = set(g["gold_families"])
        r = recs.get(key)
        if not gf or not r:
            continue
        for rd in r.get("loop_rounds") or []:
            fams_r = {fam_of.get(_canon(p), "") for p in rd.get("pool_pubs") or []}
            by_round.setdefault(rd["round"], {"hit": 0, "gold": 0, "serp": 0, "gp": 0, "queries": 0})
            by_round[rd["round"]]["hit"] += len(gf & fams_r)
            by_round[rd["round"]]["gold"] += len(gf)
            by_round[rd["round"]]["serp"] += rd.get("serpapi_calls", 0)
            by_round[rd["round"]]["gp"] += rd.get("gp_calls", 0)
            by_round[rd["round"]]["queries"] += rd.get("n_queries", 0)
    n = len(per_q)
    tot = sum(q["n_gold_fam"] for q in per_q) or 1
    with_prune = [q for q in per_q if q["pruned_hit"] is not None]
    prune_block = {}
    if with_prune:
        pt = sum(q["n_gold_fam"] for q in with_prune) or 1
        prune_block = {"family_recall_pruned": round(sum(q["pruned_hit"] for q in with_prune) / pt, 4),
                       "gold_lost_in_prune": sum(q["pool_hit"] - q["pruned_hit"] for q in with_prune),
                       "avg_pruned": round(sum(q["pruned"] for q in with_prune) / len(with_prune), 1),
                       "avg_prune_calls": round(sum(q["prune_calls"] for q in with_prune) / len(with_prune), 1)}
    return {"queries": n, "gold_families": tot, "reach_by_round": by_round,
            "family_recall@10": round(sum(q["r10"] for q in per_q) / tot, 4),
            "family_recall@30": round(sum(q["r30"] for q in per_q) / tot, 4),
            "family_recall_pool": round(sum(q["pool_hit"] for q in per_q) / tot, 4),
            **prune_block,
            "avg_serpapi_calls": round(sum(q["serpapi_calls"] for q in per_q) / max(n, 1), 1),
            "avg_gp_calls": round(sum(q["gp_calls"] for q in per_q) / max(n, 1), 1),
            "queries_with_any_hit_ranked": sum(1 for q in per_q if q["r30"]),
            "queries_with_any_hit_pool": sum(1 for q in per_q if q["pool_hit"]),
            "reach_vs_ranking": {"never_retrieved": tot - sum(q["pool_hit"] for q in per_q),
                                 "in_pool_not_top30": sum(q["pool_hit"] - q["r30"] for q in per_q),
                                 "in_top30": sum(q["r30"] for q in per_q)},
            "channel_hits": chan_hits, "channel_unique_hits": chan_unique,
            "avg_pool": round(sum(q["pool"] for q in per_q) / max(n, 1), 1), "per_query": per_q}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stage", default="gold", choices=["gold", "pipeline"])
    ap.add_argument("--limit", type=int, default=10, help="pipeline: how many gold-bearing queries to run")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--bundle", default=os.environ.get("E4_BUNDLE", ""), help="gs:// bundle to pull (Cloud Run Job)")
    ap.add_argument("--push-bundle", default="", help="gs:// path to build+upload the bundle to")
    ap.add_argument("--upload", default=os.environ.get("E4_UPLOAD", ""), help="gs:// prefix to upload results to")
    args = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.bundle:
        gcs_bundle_pull(args.bundle)
    if args.push_bundle:
        gcs_bundle_push(args.push_bundle)
        return

    pairs = sample_pairs(args.n)
    if GOLD_PATH.exists():
        gold = json.loads(GOLD_PATH.read_text())
    else:
        gold = await build_gold(pairs)
        GOLD_PATH.write_text(json.dumps(gold, indent=1))
    print_sanity(gold)
    if args.stage == "gold":
        for k, g in list(gold.items())[:5]:
            print(f"  {k}: fam={g['family_id']} prio={g['priority_date']} members={g['n_members']} "
                  f"SEA={len(g['sea_cited'])} gold={len(g['gold'])} dropped={g['dropped']}")
        return

    import llm_cache
    llm_cache.install()
    from common import load_env_yaml
    load_env_yaml()
    from patent_analyzer.recall import serpapi as sp
    sp.sync_account()
    print("serpapi keys:", [f"{q['key']} used {q['used']}/{q['cap']}" for q in sp.quota_status()] or "NONE")
    only = {k for k in os.environ.get("E4_KEYS", "").split(",") if k}
    todo = [(k, g) for k, g in gold.items() if g["gold_families"] and (not only or k in only)][:args.limit]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(k, g):
        async with sem:
            try:
                return k, await run_pipeline_one(k, g)
            except Exception as exc:
                print(f"[{k}] FAILED {type(exc).__name__}: {exc}")
                return k, None
    recs = {k: r for k, r in await asyncio.gather(*(one(k, g) for k, g in todo)) if r}
    res = await score_pipeline(gold, recs)
    (RUN_DIR / "pipeline_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_query"}, indent=1))
    print(llm_cache.summary())
    if args.upload:
        gcs_results_push(args.upload)


if __name__ == "__main__":
    asyncio.run(main())
