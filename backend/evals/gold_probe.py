#!/usr/bin/env python3
"""Minimal-recall probe: for each gold family of a case, which query shape —
built ONLY from our own decomposition vocabulary (thing / patent / named
forms, the preamble's domain, seed CPC) — brings a family member into
Google Patents' top 100, and at what rank. ≤6 SerpAPI calls per family.

    python3 evals/gold_probe.py --key US20120194631A1 --tag h1e [--max-calls 6]

Writes eval_data/runs/e4/<KEY>_goldprobe_<TAG>.{json,md}. Also reports the
gold families' CPC subclasses / groups against the seeds' top subclasses.
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

from funnel import RUN_DIR, _canon, _fams, gold_family_map  # noqa: E402

STOP = set("a an the of for and or in on to with by from is are be as at into via using based method system apparatus "
           "device thereof having provided comprising said which wherein".split())


def _stem(w: str) -> str:
    return re.sub(r"(ings?|ations?|ers?|es|s|ed|ly)$", "", w)[:6]


def _words(t: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z]{3,}", (t or "").lower()) if w not in STOP}


def vocabulary(rec: dict) -> dict:
    """Our forms per facet, with the elements they came from."""
    forms: dict[str, list[tuple[str, str]]] = {"thing": [], "patent": [], "named": [], "domain": []}
    els = rec.get("loop_elements") or []
    for e in els:
        f = e.get("facets") or {}
        for k in ("thing", "patent", "named"):
            for t in f.get(k) or []:
                forms[k].append((" ".join(str(t).lower().split()), e["id"]))
    if els:
        forms["domain"] = [(t, els[0]["id"]) for t, _ in forms["thing"] if _ == els[0]["id"]][:3]
    return forms


def ladder(gold_text: str, forms: dict, cpc_sub: str | None, cutoff: str | None) -> list[dict]:
    """Query shapes in order of increasing constraint, using forms whose words
    occur in the gold's title+abstract (the only forms that can match)."""
    gw = _words(gold_text)

    def overlap(t: str) -> int:
        return len(_words(t) & gw)
    cand = [(t, k, eid) for k in ("named", "patent", "thing") for t, eid in forms[k] if overlap(t) > 0]
    # prefer forms that overlap the gold and are rare (longer / multi-word), unique by text
    seen = set()
    ranked = []
    for t, k, eid in sorted(cand, key=lambda x: (-overlap(x[0]), -len(x[0]))):
        if t not in seen:
            seen.add(t)
            ranked.append((t, k, eid))
    dom = [t for t, _ in forms["domain"] if overlap(t) > 0][:1] or [t for t, _ in forms["domain"]][:1]

    def g(t):
        return f"({t})" if " " in t else t
    steps = []
    if ranked:
        t1 = ranked[0]
        steps.append({"shape": "1 form", "query": g(t1[0]), "forms": [t1]})
        if len(ranked) > 1:
            t2 = ranked[1]
            steps.append({"shape": "2 forms AND", "query": f"{g(t1[0])} {g(t2[0])}", "forms": [t1, t2]})
        if dom:
            steps.append({"shape": "form AND domain", "query": f"{g(t1[0])} {g(dom[0])}", "forms": [t1, (dom[0], "domain", "e0")]})
        if cpc_sub:
            # clause LAST (leading CPC=… returns 0 through SerpAPI)
            steps.append({"shape": "form AND CPC=gold group/low", "query": f"{g(t1[0])} CPC={cpc_sub}/low", "forms": [t1], "cpc": cpc_sub})
            if len(ranked) > 1:
                steps.append({"shape": "2 forms OR AND CPC", "query": f"({g(t1[0])} OR {g(ranked[1][0])}) CPC={cpc_sub}/low", "forms": [t1, ranked[1]], "cpc": cpc_sub})
    steps.append({"shape": "gold title words (upper bound, not our vocabulary)", "query": " ".join(sorted(gw)[:6]), "forms": []})
    return steps


async def probe(key: str, tag: str, max_calls: int) -> dict:
    from patent_analyzer.recall import serpapi as sp
    from patent_analyzer.recall.bigquery_patents import fetch_by_pub_nums
    gold = json.loads((RUN_DIR / "gold.json").read_text())[key]
    rec = json.loads((RUN_DIR / f"{key}_search_{tag}.json").read_text())
    fam_of = await gold_family_map(gold)
    forms = vocabulary(rec)
    cutoff = gold.get("priority_date")
    before = f"priority:{cutoff}" if cutoff else None
    gmeta = await fetch_by_pub_nums([_canon(g["pub"]) for g in gold["gold"]], with_claims=False)
    seed_cpc = (rec.get("loop_rounds") or [{}])[0].get("cpc_top") or []
    out = {"key": key, "cutoff": cutoff, "seed_cpc_top": seed_cpc, "families": []}
    calls = 0
    for g in gold["gold"]:
        m = gmeta.get(_canon(g["pub"]), {})
        cpcs = m.get("cpc_codes") or []
        subs = sorted({c[:4] for c in cpcs})
        text = (m.get("title") or g.get("title") or "") + " " + (m.get("abstract") or "")
        fam = g["family_id"]
        row = {"pub": g["pub"], "family": fam, "title": g.get("title"), "cpc_subclasses": subs, "cpc_groups": sorted({c.split("/")[0] for c in cpcs})[:6],
               "seed_cpc_overlap": sorted(set(subs) & set(seed_cpc)), "steps": []}
        # main groups, not subclasses: CPC=<subclass>/low returns nothing; prefer a group the seeds share
        groups = sorted({c.split("/")[0] for c in cpcs if len(c.split("/")[0]) >= 5})
        gsub = next((g for g in groups if g in seed_cpc), groups[0] if groups else None)
        row["cpc_group_used"] = gsub
        for step in ladder(text, forms, gsub, cutoff)[:max_calls]:
            cands, err = await sp.search_patents(step["query"], max_pages=1, before=before)
            calls += 1
            pubs = [_canon(c.pub_num) for c in cands if c.pub_num]
            rank = next((i + 1 for i, p in enumerate(pubs) if fam_of.get(p) == fam), None)
            row["steps"].append({**step, "total": sp.last_total.get(step["query"]), "returned": len(pubs), "rank": rank,
                                 "other_gold": sorted(_fams(pubs, fam_of) - {fam}), "error": err})
        out["families"].append(row)
    out["serpapi_calls"] = calls
    return out


def md(out: dict) -> str:
    L = [f"# Gold probe · {out['key']} · cutoff {out['cutoff']} · seed CPC top {out['seed_cpc_top']} · SerpAPI calls {out['serpapi_calls']}", ""]
    for r in out["families"]:
        L.append(f"## {r['pub']} (family {r['family']}) — {r['title']}")
        L.append(f"CPC subclasses {r['cpc_subclasses']} groups {r['cpc_groups']} · overlap with seed CPC: {r['seed_cpc_overlap'] or 'none'}")
        L.append("| shape | query | total | rank of gold | other gold |")
        L.append("|---|---|---:|---:|---|")
        for s in r["steps"]:
            L.append(f"| {s['shape']} | `{s['query'][:120]}` | {s['total']} | {s['rank'] if s['rank'] else '—'} | {','.join(s['other_gold']) or ''} {s['error'] or ''} |")
        L.append("")
    return "\n".join(L)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--tag", default="h1e")
    ap.add_argument("--max-calls", type=int, default=6)
    a = ap.parse_args()
    from common import load_env_yaml
    load_env_yaml()
    out = await probe(a.key, a.tag, a.max_calls)
    (RUN_DIR / f"{a.key}_goldprobe_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (RUN_DIR / f"{a.key}_goldprobe_{a.tag}.md").write_text(md(out))
    print(md(out))


if __name__ == "__main__":
    asyncio.run(main())
