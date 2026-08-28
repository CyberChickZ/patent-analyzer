#!/usr/bin/env python3
"""L1-G: what kind of gold is it, and whose stage lost it.

Harry's objection: the E4 gold is "a patent application looking for patents";
what we feed the pipeline is a paper (sometimes a manuscript). This turns that
objection into three numbers per gold family, over the h1h run (8 papers,
56 examiner-cited families):

  hit      the family is in the h1h pool
  S3-miss  we did produce an element that covers what the reference is cited
           for, and search still did not find it  -> Phase 3 failure
  S2-miss  we never produced such an element      -> Phase 2 failure,
           sub-typed so the next extraction round knows what to add

Attribution path (Harry, 2026-09-18): the reference is cited *against* the
application's own independent claims, so first locate it there
(`gold_claims` on the real claims_text from BigQuery), then match those
limitations against our extracted elements. Two signals, never merged:
te005 cosine at tau=.7 (same encoder/threshold as the Pap2Pat coverage gate,
evals/extraction_eval.embed) and an LLM judgement; disagreements are listed.

    python3 evals/gold_nature.py --stage attribute
    python3 evals/gold_nature.py --stage npl        # paper-side gold (SEA NPL)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "e4"
OUT_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature"
TAU = 0.7
TAG = os.environ.get("GOLD_NATURE_TAG", "h1h")

MISS_TYPES = ("abstraction_gap", "missing_apparatus", "missing_application",
              "missing_legal_generalization", "not_in_paper", "other")

def h1h_keys() -> list[str]:
    return sorted(p.name.split("_search_")[0] for p in RUN_DIR.glob(f"*_search_{TAG}.json"))


def load_rec(key: str) -> dict:
    return json.loads((RUN_DIR / f"{key}_search_{TAG}.json").read_text())


def load_gold() -> dict:
    return json.loads((RUN_DIR / "gold.json").read_text())


def our_elements(rec: dict) -> list[dict]:
    """Phase-2 output as the search loop saw it: [{id, text, candidate}]."""
    els = rec.get("loop_elements") or []
    return [{"id": e.get("id", ""), "text": (e.get("text") or "").strip(),
             "candidate": e.get("candidate") or "inv1"} for e in els if (e.get("text") or "").strip()]


def split_claims(claims_text: str) -> list[tuple[int, str]]:
    """BigQuery's claims_text blob -> [(claim_no, text)].

    The blob starts with a 'What is claimed is:' header and numbers claims
    '1 . A sensor fish…', so the split is on a line break followed by the
    number; anything before the first number is the header and is dropped.
    """
    out = []
    for part in re.split(r"\n\s*(?=\d{1,3}\s*\.\s+[A-Za-z])", claims_text or ""):
        m = re.match(r"^\s*(\d{1,3})\s*\.\s*(.+)$", part.strip(), re.S)
        if m and len(m.group(2).strip()) > 30:
            out.append((int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip()))
    return out


def app_claim_limitations(claims_text: str, fallback_claims: list[str] | None = None) -> list[dict]:
    """Independent claims of the application under analysis, cut into limitations.

    Same cut as the Pap2Pat coverage gate (evals/pap2pat_extraction_eval.py:94
    `gold_claims`): `_parse_claim_limitations` + `_split_preamble`. The
    dependency test is stricter than that file's `_DEPENDENT` because this
    input is the raw BigQuery blob, whose back-references come in spellings
    that regex misses ('The apparatus as set forth in  claim 3'); here any
    reference to another claim in the opening of the claim counts — the blob
    even drops the referenced number ('The apparatus as set forth in claim,
    wherein…', US20100036217A1 claim 3), so the bare word is the test.
    Returns [{lid, claim_no, text}].
    """
    from nodes.claim_mode import _parse_claim_limitations, _split_preamble

    claims = split_claims(claims_text)
    if not claims:
        claims = [(i, re.sub(r"^\s*\d+\s*\.\s*", "", c)) for i, c in enumerate(fallback_claims or [], 1)]
    out = []
    for no, text in claims:
        if re.search(r"\bclaims?\b", text[:200], re.I):
            continue  # dependent
        parsed = _parse_claim_limitations(f"{no}. {text}")
        texts = [re.sub(r"^\s*\d+\s*\.\s*", "", t) for t in _split_preamble(parsed["preamble"]) + parsed["limitations"]]
        for i, t in enumerate([t for t in texts if len(t.strip()) > 15], 1):
            out.append({"lid": f"c{no}.l{i}", "claim_no": no, "text": t.strip()})
    return out


async def load_cases() -> list[dict]:
    """One entry per h1h paper: our elements, the pool, the gold families with a
    representative document, and the application's own independent-claim
    limitations (what the examiner cited each reference against)."""
    from funnel import _canon, _fams, gold_family_map
    from patent_analyzer.recall.bigquery_patents import fetch_by_pub_nums

    gold = load_gold()
    keys = h1h_keys()
    app_meta = await fetch_by_pub_nums(keys, with_claims=True)
    gold_pubs = sorted({g["pub"] for k in keys for g in gold[k]["gold"]})
    gold_meta = await fetch_by_pub_nums(gold_pubs, with_claims=True)

    cases = []
    for key in keys:
        rec = load_rec(key)
        g = gold[key]
        fam_of = await gold_family_map(g)
        pool_pubs = {_canon(p["pub_num"]) for p in rec.get("pool", []) if p.get("pub_num")}
        reached = _fams(pool_pubs, fam_of)
        fams: dict[str, dict] = {}
        for gd in g["gold"]:
            m = gold_meta.get(_canon(gd["pub"])) or {}
            f = fams.setdefault(gd["family_id"], {"family_id": gd["family_id"], "pubs": [], "title": "",
                                                  "abstract": "", "claim1": ""})
            f["pubs"].append(gd["pub"])
            # keep the member with the most text (some family members are stubs)
            claim1 = (split_claims(m.get("claims_text", "")) or [(0, "")])[0][1]
            if len(m.get("abstract", "")) + len(claim1) > len(f["abstract"]) + len(f["claim1"]):
                f.update(title=m.get("title", "") or gd.get("title", ""),
                         abstract=m.get("abstract", ""), claim1=claim1)
            f.setdefault("title", gd.get("title", ""))
        cases.append({
            "key": key, "pair_id": g["pair_id"], "elements": our_elements(rec),
            "limitations": app_claim_limitations((app_meta.get(key) or {}).get("claims_text", "")),
            "families": list(fams.values()), "reached": reached, "pool": len(rec.get("pool", [])),
        })
    return cases


def gold_units(fam: dict) -> list[str]:
    """What the citing examiner would have read in the reference: its title +
    abstract, its claim 1 whole, and claim 1 cut into limitations."""
    from nodes.claim_mode import _parse_claim_limitations, _split_preamble

    units = []
    head = f"{fam.get('title', '')}. {fam.get('abstract', '')}".strip(". ").strip()
    if head:
        units.append(head)
    c1 = (fam.get("claim1") or "").strip()
    if c1:
        units.append(c1[:2000])
        parsed = _parse_claim_limitations(c1)
        units += [t.strip() for t in _split_preamble(parsed["preamble"]) + parsed["limitations"]
                  if len(t.strip()) > 15]
    return units[:24] or [fam.get("title", "") or fam["family_id"]]


def cosine_signal(case: dict) -> dict:
    """te005 cosines, same encoder and tau as the Pap2Pat coverage gate
    (evals/extraction_eval.embed, SEMANTIC_SIMILARITY, tau=.7).

    Two matrices, kept apart:
      lim_cov[lid]  = best cosine of that independent-claim limitation against
                      any element we extracted  -> did Phase 2 produce it?
      fam[fid]      = best cosine of the reference's own text against any
                      element  -> is the reference's subject matter in the paper?
    """
    from extraction_eval import embed

    els = case["elements"]
    if not els:
        return {"lim_cov": {}, "fam": {}}
    ev = embed([e["text"] for e in els])
    out = {"lim_cov": {}, "fam": {}}
    lims = case["limitations"]
    if lims:
        sim = embed([l["text"] for l in lims]) @ ev.T
        for i, l in enumerate(lims):
            j = int(np.argmax(sim[i]))
            out["lim_cov"][l["lid"]] = {"cos": round(float(sim[i, j]), 4), "element": els[j]["id"]}
    for fam in case["families"]:
        units = gold_units(fam)
        sim = embed(units) @ ev.T
        i, j = np.unravel_index(int(np.argmax(sim)), sim.shape)
        out["fam"][fam["family_id"]] = {"cos": round(float(sim[i, j]), 4), "element": els[j]["id"],
                                        "unit": units[i][:120]}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="attribute", choices=["claims", "cos", "attribute", "npl"])
    args = ap.parse_args()
    from common import load_env_yaml
    load_env_yaml()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "claims":
        cases = await load_cases()
        print(f"{'case':<18}{'elems':>6}{'indep lims':>11}{'gold fam':>9}{'in pool':>8}{'pool':>7}")
        for c in cases:
            print(f"{c['key']:<18}{len(c['elements']):>6}{len(c['limitations']):>11}"
                  f"{len(c['families']):>9}{len(c['reached']):>8}{c['pool']:>7}")
        tot = sum(len(c["families"]) for c in cases)
        print(f"{'total':<18}{sum(len(c['elements']) for c in cases):>6}"
              f"{sum(len(c['limitations']) for c in cases):>11}{tot:>9}"
              f"{sum(len(c['reached']) for c in cases):>8}")
        return

    if args.stage == "cos":
        cases = await load_cases()
        n_lim = n_lim_hit = n_fam = n_fam_hit = 0
        for c in cases:
            sig = cosine_signal(c)
            lim_hit = sum(1 for v in sig["lim_cov"].values() if v["cos"] >= TAU)
            fam_hit = sum(1 for v in sig["fam"].values() if v["cos"] >= TAU)
            n_lim += len(sig["lim_cov"]); n_lim_hit += lim_hit
            n_fam += len(sig["fam"]); n_fam_hit += fam_hit
            print(f"{c['key']:<18} lim>=tau {lim_hit:>2}/{len(sig['lim_cov']):<3} "
                  f"fam>=tau {fam_hit:>2}/{len(sig['fam']):<3} "
                  f"fam cos {sorted(round(v['cos'], 3) for v in sig['fam'].values())}")
        print(f"total: limitations covered {n_lim_hit}/{n_lim} = {n_lim_hit / max(n_lim, 1):.3f}  "
              f"gold families with cos>={TAU} {n_fam_hit}/{n_fam} = {n_fam_hit / max(n_fam, 1):.3f}")
        return


if __name__ == "__main__":
    asyncio.run(main())
