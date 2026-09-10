#!/usr/bin/env python3
"""N5: why the paper-side gold reach is .118 — one autopsy row per NPL gold.

Input: eval_data/runs/gold_nature/npl_h1h.json (L1-G's resolved examiner-cited
NPL) and the h1h run records. Stages:

  pool   where each of the 17 landed in our own funnel (fuzzy title match, so a
         normalisation artefact cannot be mistaken for a miss)
  search can S2 / OpenAlex return it for the queries we actually issue
  neigh  is it inside the input paper's citation neighbourhood (refs / cits /
         hop-2 refs of the input paper on S2)

    python3 evals/scratch_n5_npl_autopsy.py --stage pool
    python3 evals/scratch_n5_npl_autopsy.py --stage search
    python3 evals/scratch_n5_npl_autopsy.py --stage neigh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "e4"
OUT_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature"
TAG = "h1h"


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:120]


def load_rec(key: str) -> dict:
    return json.loads((RUN_DIR / f"{key}_search_{TAG}.json").read_text())


def golds() -> list[dict]:
    """The 17 distinct resolved NPL, same dedupe key as gold_nature.report_npl
    (resolved DOI, else the parsed title lowercased)."""
    per_key = json.loads((OUT_DIR / f"npl_{TAG}.json").read_text())
    out, seen = [], set()
    for key in sorted(per_key):
        for r in per_key[key]:
            res = r.get("resolved") or {}
            if not res:
                continue
            k = (key, (res.get("doi") or r["title"].lower()))
            if k in seen:
                continue
            seen.add(k)
            out.append({"key": key, "npl_text": r["npl_text"], "cited_title": r["title"],
                        "doi": res.get("doi", ""), "title": res.get("title") or r["title"],
                        "year": res.get("year"), "source": res.get("source"),
                        "suspect": r.get("suspect"), "in_pool_l1g": r.get("in_pool")})
    return out


def locate_in_funnel(g: dict, rec: dict) -> dict:
    """Best match of this gold paper among the run's non-patent funnel docs."""
    best, best_r = None, 0.0
    nt = _norm(g["title"])
    doi = (g["doi"] or "").lower()
    for d in rec.get("funnel_docs", []):
        if d.get("match_type") == "Patent":
            continue
        if doi and (d.get("pub_num") or "").lower() == doi:
            return {"how": "doi", "ratio": 1.0, "doc": d}
        dn = _norm(d.get("title", ""))
        if not dn:
            continue
        if dn == nt or dn.startswith(nt[:60]) or nt.startswith(dn[:60]):
            return {"how": "title", "ratio": 1.0, "doc": d}
        r = SequenceMatcher(None, nt[:80], dn[:80]).ratio()
        if r > best_r:
            best, best_r = d, r
    return {"how": "fuzzy" if best_r >= 0.80 else "miss", "ratio": round(best_r, 3),
            "doc": best if best_r >= 0.80 else None}


def landing(doc: dict | None, rec: dict) -> str:
    if doc is None:
        return "not_in_pool"
    if doc.get("rank"):
        return f"top30 #{doc['rank']}"
    nt = _norm(doc.get("title", ""))
    if any(_norm(p.get("title", "")) == nt for p in rec.get("pruned", []) if nt):
        return "pruned_kept(60)"
    if doc.get("worth_reading"):
        return "screen_worth_not_kept"
    if doc.get("stage1"):
        return "stage1_kept_screen_dropped"
    return "pool_only_stage1_cut"


def stage_pool():
    gs = golds()
    rows = []
    print(f"{'#':<3}{'case':<18}{'yr':<6}{'landing':<28}{'cos':>7}{'src':<14} title")
    for i, g in enumerate(gs, 1):
        rec = load_rec(g["key"])
        m = locate_in_funnel(g, rec)
        d = m["doc"]
        land = landing(d, rec)
        row = {**g, "match_how": m["how"], "match_ratio": m["ratio"], "landing": land,
               "cos": (d or {}).get("cos"), "pool_sources": (d or {}).get("sources"),
               "elements": (d or {}).get("elements"), "screen_reason": (d or {}).get("reason")}
        rows.append(row)
        print(f"{i:<3}{g['key']:<18}{str(g['year'] or '-'):<6}{land:<28}"
              f"{(row['cos'] if row['cos'] is not None else 0):>7.3f}"
              f"{','.join(row['pool_sources'] or ['-'])[:13]:<14} {g['title'][:58]}")
    (OUT_DIR / "n5_autopsy_pool.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    n_in = sum(1 for r in rows if r["landing"] != "not_in_pool")
    print(f"\nfuzzy-checked pool reach = {n_in}/{len(rows)} = {n_in / len(rows):.3f} "
          f"(L1-G exact-match reach = {sum(1 for r in rows if r['in_pool_l1g'])}/{len(rows)})")


def our_queries(key: str) -> list[str]:
    """The paper-side queries this run actually issued: the neighbourhood
    keyword queries (S2 + OpenAlex, <=2 per candidate)."""
    from patent_analyzer.agentic.neighbourhood import keyword_queries
    rec = load_rec(key)
    els = rec.get("loop_elements") or []
    by_cand: dict[str, list[dict]] = {}
    for e in els:
        by_cand.setdefault(e.get("candidate") or e["id"].split(".")[0], []).append(e)
    out = []
    for cid, es in list(by_cand.items())[:4]:
        for q in keyword_queries({"id": cid, "elements": es})[:2]:
            out.append(q)
    return out


async def stage_search():
    from common import load_env_yaml
    load_env_yaml()
    from patent_analyzer.recall import openalex, semantic_scholar

    gs = golds()
    per_key: dict[str, list[str]] = {}
    for g in gs:
        per_key.setdefault(g["key"], our_queries(g["key"]))
    # one search per (key, query), reused for every gold of that application
    hits: dict[tuple[str, str], list[str]] = {}
    for key, qs in per_key.items():
        for q in qs:
            s2, err1 = await semantic_scholar.search(q, limit=100)
            oa, err2 = await openalex.search_works(q, limit=100)
            hits[(key, q, "s2")] = [_norm(c.title) for c in s2]
            hits[(key, q, "oa")] = [_norm(c.title) for c in oa]
            print(f"  {key} [{q}] s2={len(s2)}{' ' + str(err1) if err1 else ''} "
                  f"oa={len(oa)}{' ' + str(err2) if err2 else ''}")
    rows = []
    for g in gs:
        nt = _norm(g["title"])
        found = []
        for q in per_key[g["key"]]:
            for ch in ("s2", "oa"):
                lst = hits.get((g["key"], q, ch), [])
                for i, t in enumerate(lst, 1):
                    if t == nt or SequenceMatcher(None, nt[:80], t[:80]).ratio() >= 0.9:
                        found.append({"channel": ch, "query": q, "rank": i})
                        break
        # title-exact lookup: can the channel find it at all when asked by name?
        try:
            byname = await semantic_scholar.match_title(g["title"])
        except Exception:
            byname = None
        rows.append({**g, "queries": per_key[g["key"]], "found_in_our_queries": found,
                     "resolvable_by_title": bool(byname and byname.title)})
        print(f"{g['key']:<18}{'FOUND ' + str(found) if found else 'not in any of our query results':<40} "
              f"{g['title'][:60]}")
    (OUT_DIR / "n5_autopsy_search.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    n = sum(1 for r in rows if r["found_in_our_queries"])
    print(f"\nreturned by at least one of the queries we actually issue: {n}/{len(rows)}")


async def stage_neigh():
    from common import load_env_yaml
    load_env_yaml()
    from openworld_eval import render_paper
    from patent_analyzer.agentic import neighbourhood as nb
    from patent_analyzer.recall import semantic_scholar as ss

    gs = golds()
    gold_json = json.loads((RUN_DIR / "gold.json").read_text())
    rows = []
    for key in sorted({g["key"] for g in gs}):
        rec = load_rec(key)
        info = (rec.get("loop_rounds") or [{}])[0].get("neighbourhood") or {}
        loc = info.get("located") or {}
        pid = loc.get("paperId")
        titles: dict[str, str] = {}
        if pid:
            refs, _ = await ss.references_all(pid, nb.MAX_REFS)
            cits, _ = await ss.citations_all(pid, nb.MAX_CITS)
            for c in refs:
                titles.setdefault(_norm(c.title), "references")
            for c in cits:
                titles.setdefault(_norm(c.title), "citations")
        for g in [x for x in gs if x["key"] == key]:
            nt = _norm(g["title"])
            where = titles.get(nt) or next(
                (v for t, v in titles.items()
                 if SequenceMatcher(None, nt[:80], t[:80]).ratio() >= 0.9), "")
            rows.append({**g, "input_paper": loc.get("title"), "hop1": where or "no",
                         "hop1_size": len(titles)})
            print(f"{key:<18}{(where or 'not in hop-1'):<14}{g['title'][:64]}")
    (OUT_DIR / "n5_autopsy_neigh.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    n = sum(1 for r in rows if r["hop1"] != "no")
    print(f"\nin the input paper's own hop-1 neighbourhood: {n}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pool", choices=["pool", "search", "neigh"])
    a = ap.parse_args()
    if a.stage == "pool":
        stage_pool()
    elif a.stage == "search":
        asyncio.run(stage_search())
    else:
        asyncio.run(stage_neigh())


if __name__ == "__main__":
    main()
