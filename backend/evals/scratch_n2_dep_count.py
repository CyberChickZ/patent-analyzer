#!/usr/bin/env python3
"""N2: how many dependent claims should the draft node write?

Reads the draft runs written by evals/draft_eval.py under one tag per cap
(DRAFT_MAX_DEPENDENTS=10/20/30 -> `*_lean+dep10.json` etc.) and reports, per
cap:

  filed          claims actually filed, and the 37 CFR 1.16(h)/(i) excess-claim
                 fee that set costs. The node mirrors every dependent onto the
                 second independent claim, so a cap of N is 2 + 2N claims.
  hard rule      MPEP 608.01(n) III: pool items the further-limitation check
                 rejects, by pool depth — the question is whether the deeper
                 items start failing it (padding) or not.
  duplicates     the same check's `duplicate` verdict (cosine >= FURTHER_TAU to
                 a dependent already kept).
  112(b) rules   patent_analyzer.draft.definiteness over the dependent claims.
  PEDANTIC       --llm: the examination prompt (evals/pedantic_definiteness_eval
                 .detect) over the dependent claims only, p_indefinite >= .5.

The depth curve is computed offline by replaying avoid.dependent_claims on the
stored pool with no cap, which is exact: the cap is the FIRST test in that
loop, so an item past the cap is never asked whether it further limits. One
replay therefore gives every cap's answer, and `--verify` checks the replay
reproduces each run's actual kept set.

    python3 evals/scratch_n2_dep_count.py --tags _lean+dep10 _lean+dep20 _lean+dep30
    python3 evals/scratch_n2_dep_count.py --tags ... --llm --limit-per-tag 30
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import load_env_yaml

RUNS = Path(__file__).parent.parent / "eval_data" / "runs" / "draft_eval"

# 37 CFR 1.16(h)/(i), USPTO fee schedule effective 2025-01-19 (last revised
# 2026-08-14), read 2026-09-18. (undiscounted, small, micro), USD per claim.
FEE_EXCESS_INDEP = (600.0, 240.0, 120.0)     # each independent claim over 3
FEE_EXCESS_TOTAL = (200.0, 80.0, 40.0)       # each claim (dependent or not) over 20


def excess_fee(n_total: int, n_indep: int) -> dict:
    over_t, over_i = max(0, n_total - 20), max(0, n_indep - 3)
    return {"excess_total": over_t, "excess_indep": over_i,
            "usd": [round(over_t * t + over_i * i, 2) for t, i in zip(FEE_EXCESS_TOTAL, FEE_EXCESS_INDEP)]}


def load_runs(tag: str) -> list[dict]:
    suffix = f"{tag}.json" if tag else ".json"
    out = []
    for f in sorted(RUNS.glob("*.json")):
        if tag and not f.name.endswith(suffix):
            continue
        if not tag and any(f.stem.endswith(x) for x in ("dep10", "dep20", "dep30")):
            continue
        rec = json.loads(f.read_text())
        if rec.get("draft", {}).get("claims"):
            out.append(rec)
    return out


def dependents_of(draft: dict, parent_no: int) -> list[dict]:
    return [c for c in draft.get("claims") or [] if c.get("depends_on") == parent_no]


def structural(recs: list[dict]) -> dict:
    tot = {"docs": len(recs), "claims": 0, "indep": 0, "dep_per_parent": [], "pool": 0, "pool_dropped": 0,
           "fee_usd": [0.0, 0.0, 0.0], "over_20": 0, "rej": {}}
    for rec in recs:
        d = rec["draft"]
        claims = d.get("claims") or []
        indep = [c for c in claims if c.get("depends_on") is None]
        tot["claims"] += len(claims)
        tot["indep"] += len(indep)
        tot["dep_per_parent"].append(len(dependents_of(d, indep[0]["no"])) if indep else 0)
        pool = (d.get("avoidance") or {}).get("pool") or []
        tot["pool"] += len(pool)
        tot["pool_dropped"] += sum(1 for p in pool if p.get("dropped"))
        f = excess_fee(len(claims), len(indep))
        tot["over_20"] += 1 if f["excess_total"] else 0
        tot["fee_usd"] = [a + b for a, b in zip(tot["fee_usd"], f["usd"])]
        for r in (d.get("avoidance") or {}).get("rejected_dependents") or []:
            k = r.get("rejected") or "?"
            tot["rej"][k] = tot["rej"].get(k, 0) + 1
    return tot


def rule_flag_rate(recs: list[dict]) -> dict:
    """patent_analyzer.draft.definiteness over every dependent claim (free)."""
    from patent_analyzer.draft import definiteness as D
    n = flagged = 0
    cats: dict[str, int] = {}
    for rec in recs:
        d = rec["draft"]
        by_no = {c.get("no"): c for c in d.get("claims") or []}
        for c in d.get("claims") or []:
            if c.get("depends_on") is None:
                continue
            parents, p = [], c.get("depends_on")
            while p is not None and p in by_no:
                parents.insert(0, by_no[p])
                p = by_no[p].get("depends_on")
            flags = D.check(c, parents)
            n += 1
            if flags:
                flagged += 1
            for f in flags:
                cats[f["category"]] = cats.get(f["category"], 0) + 1
    return {"n": n, "flagged": flagged, "rate": flagged / n if n else 0.0, "categories": cats}


def depth_curve(recs: list[dict], buckets=((1, 8), (9, 10), (11, 20), (21, 30), (31, 999))) -> dict:
    """Replay avoid.dependent_claims with no cap; verdict by pool depth."""
    from patent_analyzer.draft import assemble as A
    from patent_analyzer.draft import avoid as V
    sim = A.embed_similarity
    try:
        sim(["a"], ["a"])
    except Exception as e:
        print(f"  (embeddings unavailable: {type(e).__name__}; using difflib)", file=sys.stderr)
        sim = A.difflib_similarity
    per_pos: dict[int, dict[str, int]] = {}
    kept_by_doc = {}
    for rec in recs:
        d = rec["draft"]
        claims = d.get("claims") or []
        parent = claims[0]
        items = [p for p in (d.get("avoidance") or {}).get("pool") or [] if not p.get("dropped")]
        kept, rejected = V.dependent_claims(items, parent, sim, max_n=10_000)
        verdict = {k["pid"]: "kept" for k in kept}
        verdict.update({r["pid"]: r["rejected"] for r in rejected})
        kept_by_doc[rec["publication_number"]] = [k["pid"] for k in kept]
        depth = 0
        for it in items:
            v = verdict.get(it["pid"], "?")
            if v == "kept":
                depth += 1                      # depth = position among ACCEPTED dependents
                slot = depth
            else:
                slot = depth + 1                # it was tried at this slot and failed
            per_pos.setdefault(slot, {}).setdefault(v, 0)
            per_pos[slot][v] += 1
    # does the replay reproduce what the run actually kept? (prefix check, since
    # the run's cap truncates). It does on 4 of the 5 Dis2Pat documents; on
    # US10671841B2 it keeps one fewer, because the claim text stored in the run
    # file went through the node's 112(b) auto_fix AFTER the dependents were
    # chosen, so one cosine lands on the other side of FURTHER_TAU.
    agree = []
    for rec in recs:
        d = rec["draft"]
        parent_no = (d.get("claims") or [{}])[0].get("no")
        actual = [l.get("pid") for c in d.get("claims") or [] if c.get("depends_on") == parent_no
                  for l in c.get("limitations") or []]
        replay = kept_by_doc.get(rec["publication_number"], [])
        agree.append(replay[:len(actual)] == actual)
    out = {"buckets": [], "kept_by_doc": kept_by_doc, "replay_matches_run": f"{sum(agree)}/{len(agree)}",
           "max_kept": max((len(v) for v in kept_by_doc.values()), default=0)}
    for lo, hi in buckets:
        agg: dict[str, int] = {}
        for pos, d in per_pos.items():
            if lo <= pos <= hi:
                for k, v in d.items():
                    agg[k] = agg.get(k, 0) + v
        n = sum(agg.values())
        if not n:
            continue
        bad = agg.get("not_further_limiting", 0)
        out["buckets"].append({"range": f"{lo}-{hi if hi < 999 else ''}", "tried": n, "kept": agg.get("kept", 0),
                               "not_further_limiting": bad, "duplicate": agg.get("duplicate", 0),
                               "fail_rate": round(1 - agg.get("kept", 0) / n, 3),
                               "nfl_rate": round(bad / n, 3)})
    return out


async def pedantic(recs: list[dict], limit: int, threshold: float = 0.5) -> dict:
    """PEDANTIC examination prompt over the dependent claims of the FIRST
    independent claim, in claim order, so the per-claim verdict can be bucketed
    by the depth at which that dependent was added. The mirror-form dependents
    carry the same limitation text and are skipped (they would double the bill
    and measure the same thing)."""
    from pedantic_definiteness_eval import detect
    from patent_analyzer.draft.assemble import render_claim
    rows = []
    for rec in recs:
        d = rec["draft"]
        claims = d.get("claims") or []
        by_no = {c.get("no"): c for c in claims}
        first = next((c for c in claims if c.get("depends_on") is None), None)
        if not first:
            continue
        for depth, c in enumerate(dependents_of(d, first["no"]), 1):
            parents, p = [], c.get("depends_on")
            while p is not None and p in by_no:
                parents.insert(0, {"no": p, "text": render_claim(by_no[p])})
                p = by_no[p].get("depends_on")
            rows.append({"id": f"{rec['publication_number']}-c{c['no']}", "claim_no": c["no"], "depth": depth,
                         "claim_text": render_claim(c), "parents": parents, "description": "", "label": None})
    rows = rows[:limit]
    by_id = {r["id"]: r for r in rows}
    out = await detect(rows, threshold=threshold, use_llm=True, batch=2)   # concurrency 2: Vertex is busy
    ind = sum(r["pred_llm"] for r in out)
    cats: dict[str, int] = {}
    depth_bins: dict[str, list[int]] = {}
    for r in out:
        for c in r["llm_categories"]:
            cats[c] = cats.get(c, 0) + 1
        dep = by_id[r["id"]]["depth"]
        key = "1-8" if dep <= 8 else "9-10" if dep <= 10 else "11+"
        depth_bins.setdefault(key, []).append(r["pred_llm"])
    return {"n": len(out), "indefinite": ind, "rate": ind / len(out) if out else None,
            "rule_or_llm": sum(r["pred_union"] for r in out), "categories": cats,
            "by_depth": {k: {"n": len(v), "indefinite": sum(v), "rate": round(sum(v) / len(v), 3)}
                         for k, v in sorted(depth_bins.items())},
            "per_claim": [{"id": r["id"], "depth": by_id[r["id"]]["depth"], "p": r["p_llm"],
                           "pred": r["pred_llm"], "cats": r["llm_categories"]} for r in out]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help='run-file suffixes, e.g. _lean+dep10 (use "" for untagged)')
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--limit-per-tag", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-live-calls", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    load_env_yaml()
    if args.llm:
        from draft_eval import install_budget
        install_budget(args.max_live_calls)

    report = {}
    for tag in args.tags:
        recs = load_runs(tag)
        if not recs:
            print(f"{tag or '(untagged)'}: no runs")
            continue
        s = structural(recs)
        r = rule_flag_rate(recs)
        dc = depth_curve(recs)
        row = {"tag": tag, "structural": s, "rules_112b": r, "depth": dc}
        if args.llm:
            row["pedantic"] = await pedantic(recs, args.limit_per_tag, args.threshold)
        report[tag or "base"] = row
        deps = s["dep_per_parent"]
        print(f"\n=== {tag or '(untagged, cap 8)'} : {s['docs']} docs ===")
        print(f"  dependents per independent claim: {deps} (mean {sum(deps) / len(deps):.1f})")
        print(f"  claims filed per application: mean {s['claims'] / s['docs']:.1f}  independents {s['indep'] / s['docs']:.1f}  "
              f"over 20 claims: {s['over_20']}/{s['docs']}")
        print(f"  37 CFR 1.16 excess fee, total over {s['docs']} apps (undiscounted / small / micro): "
              f"${s['fee_usd'][0]:,.0f} / ${s['fee_usd'][1]:,.0f} / ${s['fee_usd'][2]:,.0f}  "
              f"(per app ${s['fee_usd'][0] / s['docs']:,.0f} / ${s['fee_usd'][1] / s['docs']:,.0f} / ${s['fee_usd'][2] / s['docs']:,.0f})")
        print(f"  pool: {s['pool'] / s['docs']:.1f} items/doc, {s['pool_dropped']} dropped unsupported; rejections {s['rej']}")
        print(f"  112(b) rule flags on dependents: {r['flagged']}/{r['n']} = {r['rate']:.3f}  {r['categories'] or ''}")
        if args.llm:
            p = row["pedantic"]
            print(f"  PEDANTIC indefinite (p>={args.threshold}) on claim-1 dependents: {p['indefinite']}/{p['n']} = "
                  f"{p['rate']:.3f}  rules-or-LLM {p['rule_or_llm']}/{p['n']}  {p['categories']}")
            print(f"    by depth: " + "  ".join(f"{k}: {v['indefinite']}/{v['n']}={v['rate']:.3f}"
                                                for k, v in p["by_depth"].items()))
        print("  MPEP 608.01(n) III by pool depth (one uncapped replay; 'tried' = items the check actually reached):")
        print(f"    {'depth':<10}{'tried':>7}{'kept':>7}{'not_further':>13}{'dup':>6}{'fail_rate':>11}")
        for b in dc["buckets"]:
            print(f"    {b['range']:<10}{b['tried']:>7}{b['kept']:>7}{b['not_further_limiting']:>13}{b['duplicate']:>6}{b['fail_rate']:>11.3f}")
        print(f"    uncapped kept per doc: {[len(v) for v in dc['kept_by_doc'].values()]} (max {dc['max_kept']}); "
              f"replay reproduces the run's kept set on {dc['replay_matches_run']} docs")

    if args.llm:
        import app.llm as llm
        import llm_cache
        from patent_analyzer import metering
        cost = 0.0
        for model, u in (llm.usage or {}).items():
            pin, pout = metering.PRICES.get(model, (0.0, 0.0))
            # thought tokens bill as output (metering.py rate-card note)
            cost += (u.get("prompt_tokens", 0) * pin
                     + (u.get("output_tokens", 0) + u.get("thought_tokens", 0)) * pout) / 1e6
        print(f"\nlive LLM calls {llm_cache.stats['misses']}, cached {llm_cache.stats['hits']}; "
              f"estimated spend for this script ${cost:.3f} ({dict(llm.usage or {})})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    asyncio.run(main())
