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

PAP2PAT = Path("/tmp/pap2pat/Pap2Pat/data")
RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "e4"
GOLD_PATH = RUN_DIR / "gold.json"


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


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stage", default="gold", choices=["gold"])
    args = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    pairs = sample_pairs(args.n)
    if GOLD_PATH.exists():
        gold = json.loads(GOLD_PATH.read_text())
    else:
        gold = await build_gold(pairs)
        GOLD_PATH.write_text(json.dumps(gold, indent=1))
    print_sanity(gold)
    for k, g in list(gold.items())[:5]:
        print(f"  {k}: fam={g['family_id']} prio={g['priority_date']} members={g['n_members']} "
              f"SEA={len(g['sea_cited'])} gold={len(g['gold'])} dropped={g['dropped']}")


if __name__ == "__main__":
    asyncio.run(main())
