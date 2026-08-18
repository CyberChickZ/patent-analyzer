#!/usr/bin/env python3
"""Per-case search funnel: what every call brought back and where the gold went.

Reads one E4 record (eval_data/runs/e4/<KEY>_search_<TAG>.json) plus gold.json
and writes <KEY>_funnel_<TAG>.{json,md}: one row per query / expansion /
prune stage / rerank with returned, new, gold-family and cumulative-reach
columns. Records written before the per-query id logging (82c1260) are
reconstructed: the wide queries are rebuilt from the stored facets and
looked up in the KV search cache; expansion is re-derived from BigQuery
citations of the seeds (ids only).

    python3 evals/funnel.py --tag h1d                # every record with that tag
    python3 evals/funnel.py --tag h1d --key US20120194631A1
    python3 evals/funnel.py --tag h1d --summary      # H6 per-kind gold yield table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "e4"


def _canon(p: str) -> str:
    p = re.sub(r"[\s\-/,.]", "", (p or "").upper())
    m = re.match(r"^US(\d{10})([A-Z]\d?)?$", p)
    if m and m.group(1)[:2] in ("19", "20"):
        return f"US{m.group(1)[:4]}0{m.group(1)[4:]}{m.group(2) or ''}"
    return p


async def gold_family_map(gold_entry: dict) -> dict[str, str]:
    """pub → gold family id for every member of every gold family."""
    from patent_analyzer.recall.bigquery_patents import fetch_families
    fams = await fetch_families(gold_entry["gold_families"])
    fam_of = {_canon(m["publication_number"]): fid for fid, ms in fams.items() for m in ms}
    for gd in gold_entry["gold"]:
        fam_of.setdefault(_canon(gd["pub"]), gd["family_id"])
    return fam_of


def _fams(pubs, fam_of) -> set[str]:
    return {fam_of[_canon(p)] for p in pubs if _canon(p) in fam_of}


async def _reconstruct_queries(rec: dict, cutoff: str | None) -> list[dict]:
    """Old records: rebuild the wide queries from the stored facets and read
    the KV search cache for their result ids."""
    from patent_analyzer.agentic.wide import wide_queries
    from patent_analyzer.cache import kv
    from patent_analyzer.recall import google_patents as gp
    from patent_analyzer.recall import serpapi as sp

    els = rec.get("loop_elements") or (rec.get("loop_rounds") or [{}])[0].get("elements") or []
    stats_els = els or []
    by_cand: dict[str, list[dict]] = {}
    for e in stats_els:
        by_cand.setdefault(e.get("candidate") or "inv1", []).append(e)
    cands = [{"id": cid, "elements": es} for cid, es in by_cand.items()]
    out = []
    before = f"priority:{cutoff}" if cutoff else None
    for i, q in enumerate(wide_queries(cands), 1):
        hit = kv().get("search", sp._cache_key("google_patents", q["query"], 1, 100, before, True), max_age_days=60)
        chan = "serpapi_patents"
        if hit is None:
            hit = kv().get("search", gp._cache_key(q["query"], 100, 0, before), max_age_days=60)
            chan = "google_patents"
        pubs = [c.get("pub_num") or (c.get("title") or "")[:80] for c in (hit or {}).get("cands", [])]
        out.append({"n": i, "candidate": q["candidate"], "kind": q["kind"], "query": q["query"],
                    "facets_used": q.get("facets_used", {}), "elements": q.get("elements", []),
                    "channel": chan if hit else "cache-miss", "total": (hit or {}).get("total"), "hits": len(pubs),
                    "pubs": pubs, "papers": sum(1 for c in (hit or {}).get("cands", []) if c.get("match_type") != "Patent"),
                    "reconstructed": True})
    return out


async def _reconstruct_expansion(seed_pubs: list[str], pool_pubs: set[str], cutoff: str | None) -> dict[str, list[str]]:
    from patent_analyzer.recall.bigquery_patents import fetch_citations
    cits = await fetch_citations(seed_pubs)
    out = {}
    for s, c in cits.items():
        kept = [_canon(x["cited"]) for x in c.get("cits", []) if not x.get("npl_text") and _canon(x["cited"]) in pool_pubs]
        if kept:
            out[_canon(s)] = list(dict.fromkeys(kept))
    return out


async def build_funnel(rec: dict, gold_entry: dict) -> dict:
    fam_of = await gold_family_map(gold_entry)
    gold_f = set(gold_entry["gold_families"])
    cutoff = gold_entry.get("priority_date")
    rd = (rec.get("loop_rounds") or [{}])[0]
    queries = rd.get("queries") or []
    if queries and "pubs" not in queries[0]:
        queries = await _reconstruct_queries(rec, cutoff)
    pool_pubs = {_canon(p["pub_num"]) for p in rec.get("pool", []) if p.get("pub_num")}
    rows, seen, reached = [], set(), set()
    # 1) queries
    for q in queries:
        pubs = [_canon(p) for p in q.get("pubs", [])]
        new = [p for p in pubs if p not in seen]
        seen.update(pubs)
        gf = _fams(pubs, fam_of)
        gnew = gf - reached
        reached |= gf
        rows.append({"stage": "query", "n": q.get("n"), "kind": q.get("kind"), "candidate": q.get("candidate"),
                     "query": q.get("query", ""), "elements": q.get("elements", []), "facets_used": q.get("facets_used", {}),
                     "channel": q.get("channel"), "total": q.get("total"), "returned": len(pubs), "new": len(new),
                     "papers": q.get("papers", 0), "gold_families": sorted(gf), "gold_new": len(gnew),
                     "reach": len(reached), "gold_pubs": [p for p in pubs if p in fam_of]})
    # 1b) paper neighbourhood → bridge (patents citing those papers)
    if rd.get("neighbourhood") is not None:
        bp = [_canon(p) for p in rd.get("bridge_pubs", [])]
        bg = _fams(bp, fam_of)
        rows.append({"stage": "bridge", "papers": rd.get("neighbourhood_papers"), "oa_ids": (rd.get("bridge") or {}).get("oa_ids"),
                     "returned": len(bp), "new": len([p for p in bp if p not in seen]), "gold_families": sorted(bg),
                     "gold_new": len(bg - reached), "reach": len(reached | bg), "error": (rd.get("bridge") or {}).get("error"),
                     "neigh_sources": (rd.get("neighbourhood") or {}).get("sources"), "neigh_seconds": (rd.get("neighbourhood") or {}).get("seconds"),
                     "gold_pubs": [p for p in bp if p in fam_of]})
        reached |= bg
        seen.update(bp)
    # 1c) Google's semantic neighbours of the seeds
    if rd.get("similar_pubs") is not None:
        sp = [_canon(p) for p in rd.get("similar_pubs", [])]
        sg = _fams(sp, fam_of)
        rows.append({"stage": "google_similar", "seeds": rd.get("seeds"), "similar_total": rd.get("similar_total"),
                     "returned": len(sp), "new": len([p for p in sp if p not in seen]), "gold_families": sorted(sg),
                     "gold_new": len(sg - reached), "reach": len(reached | sg), "gold_pubs": [p for p in sp if p in fam_of]})
        reached |= sg
        seen.update(sp)
    # 2) expansion
    by_seed = rd.get("cited_by_seed")
    expanded = set(_canon(p) for p in rd.get("expanded_pubs", []))
    if by_seed is None and rd.get("seed_pubs"):
        by_seed = await _reconstruct_expansion(rd["seed_pubs"], pool_pubs, cutoff)
        expanded = {p for ps in by_seed.values() for p in ps} if by_seed else set()
    exp_g = _fams(expanded, fam_of)
    gold_by_seed = {s: sorted(_fams(ps, fam_of)) for s, ps in (by_seed or {}).items() if _fams(ps, fam_of)}
    new_exp = exp_g - reached
    reached |= exp_g
    rows.append({"stage": "expansion", "seeds": rd.get("seeds"), "cited_total": rd.get("cited_total"),
                 "returned": len(expanded), "new": len(expanded - seen), "gold_families": sorted(exp_g),
                 "gold_new": len(new_exp), "reach": len(reached), "gold_by_seed": gold_by_seed,
                 "reconstructed": rd.get("cited_by_seed") is None})
    seen |= expanded
    # 3) other channels (pool minus loop)
    other = pool_pubs - seen
    og = _fams(other, fam_of)
    rows.append({"stage": "other_channels", "returned": len(other), "new": len(other), "gold_families": sorted(og),
                 "gold_new": len(og - reached), "reach": len(reached | og)})
    reached |= og
    pool_g = _fams(pool_pubs, fam_of)
    # 4) prune
    fd = rec.get("funnel_docs") or []
    pr = rec.get("prune") or {}
    if fd:
        s1 = [d for d in fd if d.get("stage1")]
        s2 = [d for d in fd if d.get("worth_reading")]
        rows.append({"stage": "prune_stage1_embed", "in": pr.get("stage1_in"), "out": len(s1), "cut_cos": pr.get("stage1_cut_cos"),
                     "gold_in": sorted(pool_g), "gold_out": sorted(_fams([d["pub_num"] for d in s1], fam_of)),
                     "gold_lost": [{"pub": d["pub_num"], "cos": d["cos"], "best_element": d["best_element"]}
                                   for d in fd if d.get("pub_num") and _canon(d["pub_num"]) in fam_of and not d.get("stage1")]})
        rows.append({"stage": "prune_stage2_llm", "in": pr.get("stage2_in"), "calls": pr.get("stage2_calls"), "worth": pr.get("stage2_worth"),
                     "out": pr.get("stage2_out"), "gold_out": sorted(_fams([d["pub_num"] for d in s2], fam_of)),
                     "gold_verdicts": [{"pub": d["pub_num"], "worth_reading": d.get("worth_reading"), "elements": d.get("elements"), "reason": d.get("reason")}
                                       for d in fd if d.get("pub_num") and _canon(d["pub_num"]) in fam_of and d.get("stage1")]})
    elif pr:
        pruned = [p["pub_num"] for p in rec.get("pruned", [])]
        rows.append({"stage": "prune", "in": pr.get("pool"), "stage1_out": pr.get("stage1_out"), "calls": pr.get("stage2_calls"),
                     "worth": pr.get("stage2_worth"), "out": len(pruned), "gold_in": sorted(pool_g),
                     "gold_out": sorted(_fams(pruned, fam_of)), "reconstructed": True})
    # 5) rerank
    ranked = [(i + 1, r.get("pub_num", "")) for i, r in enumerate(rec.get("ranked", []))]
    rows.append({"stage": "rerank", "out": len(ranked),
                 "gold_ranks": [{"rank": i, "pub": p, "family": fam_of[_canon(p)]} for i, p in ranked if _canon(p) in fam_of]})
    return {"key": rec.get("key"), "cutoff": cutoff, "gold_families": sorted(gold_f), "n_gold": len(gold_f),
            "reach_pool": len(pool_g), "rows": rows}


def funnel_md(f: dict) -> str:
    L = [f"# Funnel · {f['key']} · cutoff {f['cutoff']} · gold families {f['n_gold']} · in pool {f['reach_pool']}", ""]
    L.append("| # | stage | kind | candidate | elements | total | returned | new | gold fam (new) | reach | query |")
    L.append("|---|---|---|---|---|---:|---:|---:|---|---:|---|")
    for r in f["rows"]:
        if r["stage"] == "query":
            gf = ",".join(r["gold_families"]) or "—"
            L.append(f"| {r['n']} | query | {r['kind']} | {r['candidate']} | {' '.join(r['elements'])} | {r['total'] if r['total'] is not None else '?'} | "
                     f"{r['returned']} | {r['new']} | {gf} (+{r['gold_new']}) | {r['reach']} | `{r['query'][:110]}` |")
        elif r["stage"] == "bridge":
            L.append(f"| — | paper neighbourhood → bridge | {r['papers']} papers, {r['oa_ids']} OpenAlex ids | | | {r.get('neigh_seconds')}s | {r['returned']} | {r['new']} | "
                     f"{','.join(r['gold_families']) or '—'} (+{r['gold_new']}) | {r['reach']} | {r.get('error') or ''} sources={r.get('neigh_sources')} |")
        elif r["stage"] == "google_similar":
            L.append(f"| — | google similar | {r['seeds']} seeds → {r['similar_total']} neighbours | | | | {r['returned']} | {r['new']} | "
                     f"{','.join(r['gold_families']) or '—'} (+{r['gold_new']}) | {r['reach']} | shared-by-seeds ranking, date-filtered |")
        elif r["stage"] == "expansion":
            gs = "; ".join(f"{s}→{','.join(v)}" for s, v in list(r["gold_by_seed"].items())[:6]) or "—"
            L.append(f"| — | expansion | seeds {r['seeds']} | | | {r['cited_total']} cited | {r['returned']} | {r['new']} | "
                     f"{','.join(r['gold_families']) or '—'} (+{r['gold_new']}) | {r['reach']} | {gs} |")
        elif r["stage"] == "other_channels":
            L.append(f"| — | other channels | | | | | {r['returned']} | {r['new']} | {','.join(r['gold_families']) or '—'} (+{r['gold_new']}) | {r['reach']} | S2/OpenAlex/arXiv/BQ |")
        elif r["stage"] == "prune_stage1_embed":
            lost = "; ".join(f"{d['pub']} cos={d['cos']:.2f} ({d['best_element']})" for d in r["gold_lost"]) or "none"
            L.append(f"| — | prune: embedding | top-100/element | | | | {r['in']} | → {r['out']} | gold {len(r['gold_in'])}→{len(r['gold_out'])} | | cut cos {r['cut_cos']:.3f}; lost: {lost} |")
        elif r["stage"] == "prune_stage2_llm":
            v = "; ".join(f"{d['pub']}: {'keep' if d['worth_reading'] else 'DROP'} [{','.join(d['elements'] or [])}] {d['reason']}" for d in r["gold_verdicts"]) or "none in shortlist"
            L.append(f"| — | prune: LLM screen | {r['calls']} calls | | | | {r['in']} | → {r['worth']} worth → {r['out']} | gold → {len(r['gold_out'])} | | {v} |")
        elif r["stage"] == "prune":
            L.append(f"| — | prune (old record) | {r['calls']} calls | | | | {r['in']} → embed {r['stage1_out']} | → {r['worth']} → {r['out']} | gold {len(r['gold_in'])}→{len(r['gold_out'])} | | per-doc verdicts not logged in this run |")
        elif r["stage"] == "rerank":
            gr = ", ".join(f"#{d['rank']} {d['pub']}" for d in r["gold_ranks"]) or "no gold in top-30"
            L.append(f"| — | rerank | top-{r['out']} | | | | | | | | {gr} |")
    return "\n".join(L) + "\n"


def summary_table(funnels: list[dict]) -> str:
    """Gold yield per call kind across cases (H6 §4)."""
    by = {}
    for f in funnels:
        first_hit = {}
        for r in f["rows"]:
            if r["stage"] == "query":
                b = by.setdefault(r["kind"], Counter())
                b["calls"] += 1
                b["returned"] += r["returned"]
                b["new"] += r["new"]
                b["gold_any"] += len(r["gold_families"])
                b["gold_new"] += r["gold_new"]
                b["calls_with_gold"] += 1 if r["gold_families"] else 0
            elif r["stage"] in ("expansion", "other_channels", "bridge", "google_similar"):
                b = by.setdefault(r["stage"], Counter())
                b["calls"] += 1
                b["returned"] += r["returned"]
                b["new"] += r["new"]
                b["gold_any"] += len(r["gold_families"])
                b["gold_new"] += r["gold_new"]
                b["calls_with_gold"] += 1 if r["gold_families"] else 0
    L = ["| call kind | calls | returned | new | calls with any gold | gold families (any) | gold families first reached here |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k, b in sorted(by.items(), key=lambda kv: -kv[1]["gold_new"]):
        L.append(f"| {k} | {b['calls']} | {b['returned']} | {b['new']} | {b['calls_with_gold']} | {b['gold_any']} | **{b['gold_new']}** |")
    return "\n".join(L) + "\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    gold = json.loads((RUN_DIR / "gold.json").read_text())
    files = sorted(RUN_DIR.glob(f"*_search_{args.tag}.json"))
    if args.key:
        files = [f for f in files if f.name.startswith(args.key)]
    funnels = []
    for f in files:
        rec = json.loads(f.read_text())
        g = gold.get(rec["key"])
        if not g or not g.get("gold_families"):
            continue
        fn = await build_funnel(rec, g)
        funnels.append(fn)
        (RUN_DIR / f"{rec['key']}_funnel_{args.tag}.json").write_text(json.dumps(fn, ensure_ascii=False, indent=1))
        (RUN_DIR / f"{rec['key']}_funnel_{args.tag}.md").write_text(funnel_md(fn))
        print(f"{rec['key']}: gold {fn['n_gold']} in pool {fn['reach_pool']} → {RUN_DIR / (rec['key'] + '_funnel_' + args.tag + '.md')}")
    if args.summary and funnels:
        print(summary_table(funnels))


if __name__ == "__main__":
    asyncio.run(main())
