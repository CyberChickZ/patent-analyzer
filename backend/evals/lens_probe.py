#!/usr/bin/env python3
"""Offline Lens.org probe on one E4 record (K2): does Lens bring gold the loop
did not have?

  (1) bridge: the record's paper neighbourhood (loop_rounds[0].neigh_oa_ids +
      the DOIs of match_type=Paper pool entries) -> Lens scholarly records ->
      patent_citations -> Lens patent records -> pub numbers -> gold families
  (2) search: Lens Patent API bool queries from the loop's thing facets
      (kind=thing queries, first 6 terms) x cpc_top[:2] (+ no-cpc), cutoff =
      gold priority_date; <= 10 calls.

Per call: returned, gold families hit, unique contribution vs the record's
pool (pub not in pool / no simple-family member in pool).  Writes
eval_data/runs/e4/<KEY>_lens_probe.{json,md}.  Does not touch the loop.

    python3 evals/lens_probe.py --key US20120194631A1 --tag h1e
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import load_env_yaml  # noqa: E402
from funnel import RUN_DIR, _canon, gold_family_map  # noqa: E402
from patent_analyzer.recall import lens  # noqa: E402

MAX_SEARCH_CALLS = 10


def _load_record(key: str, tag: str) -> tuple[dict, str]:
    for t in (tag, "h1d"):
        p = RUN_DIR / f"{key}_search_{t}.json"
        if p.exists():
            return json.load(open(p)), t
    raise SystemExit(f"no record for {key} ({tag} / h1d)")


def _paper_ids(rec: dict) -> tuple[list[str], list[str]]:
    lr = (rec.get("loop_rounds") or [{}])[0]
    oa = list(dict.fromkeys(lr.get("neigh_oa_ids") or []))
    dois = []
    for p in rec.get("pool") or []:
        if p.get("match_type") == "Paper" and (p.get("pub_num") or "").lower().startswith("10."):
            dois.append(p["pub_num"])
    loc = ((lr.get("neighbourhood") or {}).get("located") or {})
    if loc.get("doi"):
        dois.append(loc["doi"])
    return list(dict.fromkeys(dois)), oa


def _thing_sets(rec: dict) -> list[list[str]]:
    lr = (rec.get("loop_rounds") or [{}])[0]
    sets: list[list[str]] = []
    for q in lr.get("queries") or []:
        if q.get("kind") == "thing":
            terms = list((q.get("facets_used") or {}).get("thing") or [])[:6]
            if terms and terms not in sets:
                sets.append(terms)
    if sets:
        return sets
    # older records: the per-element facets
    for e in rec.get("loop_elements") or lr.get("elements") or []:
        terms = list((e.get("facets") or {}).get("thing") or [])[:6]
        if terms and terms not in sets:
            sets.append(terms)
    return sets[:3]


def _pool_pubs(rec: dict) -> set[str]:
    return {_canon(p.get("pub_num") or "") for p in rec.get("pool") or [] if p.get("match_type") == "Patent" and p.get("pub_num")}


def _hits(cands, fam_of, pool_pubs, cutoff: str = "") -> dict:
    gold_fams, new_pubs, new_fams, rows = set(), [], [], []
    for c in cands:
        pubs = [_canon(c.pub_num)] + [_canon(m) for m in c.raw["lens"].get("family") or []]
        fams = {fam_of[p] for p in pubs if p in fam_of}
        in_pool_pub = _canon(c.pub_num) in pool_pubs
        in_pool_fam = any(p in pool_pubs for p in pubs)
        gold_fams |= fams
        if not in_pool_pub:
            new_pubs.append(c.pub_num)
        if not in_pool_fam:
            new_fams.append(c.pub_num)
        fed = c.raw["lens"].get("family_earliest_date") or ""
        rows.append({"pub": c.pub_num, "title": c.title[:90], "date": c.raw["lens"].get("date_published"),
                     "fam_earliest": fed, "before_cutoff": bool(cutoff and fed and fed.replace("-", "") < cutoff),
                     "gold_fams": sorted(fams), "in_pool_pub": in_pool_pub, "in_pool_fam": in_pool_fam})
    return {"gold_fams": sorted(gold_fams), "new_pubs": new_pubs, "new_fams": new_fams, "rows": rows}


async def main(key: str, tag: str) -> None:
    load_env_yaml()
    rec, used_tag = _load_record(key, tag)
    gold_all = json.load(open(RUN_DIR / "gold.json"))
    gold = gold_all[key]
    fam_of = await gold_family_map(gold)
    gold_fam_ids = sorted({g["family_id"] for g in gold["gold"]})
    pool_pubs = _pool_pubs(rec)
    cutoff = gold["priority_date"]
    lr = (rec.get("loop_rounds") or [{}])[0]
    out: dict = {"key": key, "tag": used_tag, "cutoff": cutoff, "gold_families": gold_fam_ids,
                 "pool_patents": len(pool_pubs)}

    # ---- (1) bridge -------------------------------------------------------
    dois, oa_ids = _paper_ids(rec)
    t0 = time.time()
    n_calls0 = len(lens.call_log)
    sch, err1 = await lens.scholarly_by_ids(dois, oa_ids)
    matched = {r["lens_id"]: r for r in sch.values() if r.get("lens_id")}
    pat_ids = sorted({pid for r in matched.values() for pid in r["patent_citations"]})
    cands, err2 = await lens.patents_by_lens_ids(pat_ids)
    h = _hits(cands, fam_of, pool_pubs, cutoff)
    by_paper = {k: {"lens_id": v["lens_id"], "title": v["title"][:80], "n_pat": v["patent_citations_count"]}
                for k, v in sch.items() if v["patent_citations_count"]}
    gold_cands = [r for r in h["rows"] if r["gold_fams"]]
    out["bridge"] = {"input_dois": len(dois), "input_oa_ids": len(oa_ids), "papers_found": len(matched),
                     "papers_with_patent_citations": sum(1 for r in matched.values() if r["patent_citations_count"]),
                     "patent_lens_ids": len(pat_ids), "patents_returned": len(cands),
                     "gold_fams_hit": h["gold_fams"], "new_pubs": len(h["new_pubs"]), "new_fams": len(h["new_fams"]),
                     "gold_rows": gold_cands, "new_gold_pubs": [r["pub"] for r in gold_cands if not r["in_pool_fam"]],
                     "calls": len(lens.call_log) - n_calls0, "seconds": round(time.time() - t0, 1),
                     "errors": [e for e in (err1, err2) if e], "papers_with_citations": by_paper,
                     "sample_new_fams": h["new_fams"][:15],
                     "patents_family_before_cutoff": sum(1 for r in h["rows"] if r["before_cutoff"])}
    # loop's own bridge (BigQuery pcs_oa) for comparison
    out["loop_bridge"] = {"stats": lr.get("bridge"), "pubs": len(lr.get("bridge_pubs") or [])}
    lb = {_canon(p) for p in lr.get("bridge_pubs") or []}
    out["bridge"]["overlap_with_loop_bridge_pubs"] = sum(1 for c in cands if _canon(c.pub_num) in lb)

    # ---- (2) search -------------------------------------------------------
    thing_sets = _thing_sets(rec)
    cpcs = list(lr.get("cpc_top") or [])[:2]
    cpc_src = "loop_rounds[0].cpc_top"
    if not cpcs:
        cnt = Counter(s[:4] for c in cands for s in c.raw["lens"].get("cpc") or [])
        cpcs = [s for s, _ in cnt.most_common(2)]
        cpc_src = "top subclasses of the Lens bridge patents' CPC (record has no cpc_top)"
    plan: list[tuple[list[str], str | None]] = []
    for ts in thing_sets:
        for c in cpcs:
            plan.append((ts, c))
    for ts in thing_sets:
        plan.append((ts, None))
    plan = plan[:MAX_SEARCH_CALLS]
    search_rows, cum_fams, cum_new_fams = [], set(), set()
    t0 = time.time()
    for i, (ts, cpc) in enumerate(plan, 1):
        n0 = len(lens.call_log)
        cs, err = await lens.search_patents(ts, cpc=cpc, before=cutoff, size=100)
        log = lens.call_log[n0:] or [{}]
        hh = _hits(cs, fam_of, pool_pubs, cutoff)
        new_gold = sorted(set(hh["gold_fams"]) - cum_fams)
        cum_fams |= set(hh["gold_fams"])
        cum_new_fams |= set(hh["new_fams"])
        search_rows.append({"n": i, "terms": ts, "cpc": cpc, "total": log[-1].get("total"), "returned": len(cs),
                            "seconds": log[-1].get("seconds"), "http_status": log[-1].get("http_status"),
                            "cached": log[-1].get("cached"),
                            "gold_fams": hh["gold_fams"], "gold_new_cum": new_gold,
                            "new_pubs": len(hh["new_pubs"]), "new_fams": len(hh["new_fams"]),
                            "gold_rows": [r for r in hh["rows"] if r["gold_fams"]], "err": err})
    out["search"] = {"cpc": cpcs, "cpc_source": cpc_src, "thing_sets": thing_sets, "calls": len(plan),
                     "seconds": round(time.time() - t0, 1), "rows": search_rows,
                     "gold_fams_hit": sorted(cum_fams), "unique_new_fams": len(cum_new_fams)}
    out["call_log"] = lens.call_log
    out["totals"] = {"calls": len(lens.call_log), "cached": sum(1 for c in lens.call_log if c.get("cached")),
                     "seconds": round(sum(c.get("seconds") or 0 for c in lens.call_log), 1),
                     "gold_fams_hit_any": sorted(set(out["bridge"]["gold_fams_hit"]) | cum_fams),
                     "gold_fams_total": len(gold_fam_ids)}

    jp = RUN_DIR / f"{key}_lens_probe.json"
    jp.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (RUN_DIR / f"{key}_lens_probe.md").write_text(_md(out))
    print(_md(out))


def _md(o: dict) -> str:
    b, s = o["bridge"], o["search"]
    L = [f"# Lens probe — {o['key']} (record tag {o['tag']}, cutoff {o['cutoff']})", "",
         f"gold families: {len(o['gold_families'])} · pool patents in record: {o['pool_patents']}", "",
         "## (1) bridge: papers → citing patents (Lens scholarly patent_citations)", "",
         f"- input: {b['input_dois']} DOIs + {b['input_oa_ids']} OpenAlex ids → {b['papers_found']} Lens papers, "
         f"{b['papers_with_patent_citations']} with patent citations → {b['patent_lens_ids']} patent lens_ids → "
         f"{b['patents_returned']} patent records ({b['calls']} calls, {b['seconds']}s)",
         f"- gold families hit: {b['gold_fams_hit'] or '—'}; new gold pubs vs pool: {b['new_gold_pubs'] or '—'}",
         f"- unique contribution vs pool: {b['new_pubs']} pubs / {b['new_fams']} families not in pool; "
         f"{b['patents_family_before_cutoff']} of {b['patents_returned']} have a family member published before cutoff",
         f"- overlap with the loop's BigQuery bridge pubs: {b['overlap_with_loop_bridge_pubs']} "
         f"(loop bridge: {o['loop_bridge']['stats']})",
         f"- errors: {b['errors'] or '—'}", ""]
    if b["gold_rows"]:
        L += ["| pub | title | date | family earliest | before cutoff | gold fam | in pool (pub/fam) |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r['pub']} | {r['title']} | {r['date']} | {r['fam_earliest']} | {r['before_cutoff']} | {','.join(r['gold_fams'])} | {r['in_pool_pub']}/{r['in_pool_fam']} |" for r in b["gold_rows"]]
        L.append("")
    L += ["## (2) search: Patent API bool queries", "",
          f"cpc: {s['cpc']} ({s['cpc_source']}); {s['calls']} calls, {s['seconds']}s; "
          f"gold families hit: {s['gold_fams_hit'] or '—'}; unique new families vs pool: {s['unique_new_fams']}", "",
          "| # | terms | cpc | total | returned | s | gold fams | new gold (cum) | new pubs | new fams | err |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        L.append(f"| {r['n']} | {'; '.join(r['terms'])[:70]} | {r['cpc'] or '—'} | {r['total']} | {r['returned']} | "
                 f"{r['seconds']}{' (cache)' if r['cached'] else ''} | {','.join(r['gold_fams']) or '—'} | "
                 f"{','.join(r['gold_new_cum']) or '—'} | {r['new_pubs']} | {r['new_fams']} | {r['err'] or ''} |")
    gr = [(r["n"], g) for r in s["rows"] for g in r["gold_rows"]]
    if gr:
        L += ["", "gold rows from search:", ""] + [f"- q{n}: {g['pub']} — {g['title']} ({g['date']}) fam {','.join(g['gold_fams'])}, in pool fam={g['in_pool_fam']}" for n, g in gr]
    t = o["totals"]
    L += ["", "## totals", "",
          f"- Lens calls: {t['calls']} ({t['cached']} cached), {t['seconds']}s network",
          f"- gold families hit by any Lens path: {len(t['gold_fams_hit_any'])}/{t['gold_fams_total']} {t['gold_fams_hit_any']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--tag", default="h1e")
    a = ap.parse_args()
    asyncio.run(main(a.key, a.tag))
