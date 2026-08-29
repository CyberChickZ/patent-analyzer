"""M1 round controller: run the moves, read the claims, keep what is GOOD.

Round 0 starts from the input paper alone (predicted CPC enumeration, the
paper's own neighbourhood bridge, and the ReAct queries). Every later round
starts from the GOOD set that the claims judge produced, which is the whole
point: the old loop expanded from every query hit, this one expands only from
documents whose claims a model could point at.
"""

from __future__ import annotations

import asyncio
import os
import time

from . import good as G
from . import moves as M
from ..recall.pool import Candidate

CLAIMS_CHUNK = int(os.environ.get("M1_CLAIMS_CHUNK", "300"))


async def _claims_for(cands: list[Candidate]) -> tuple[dict[str, str], int]:
    """Claims text for a round's candidates, chunked so no BigQuery query goes
    near the 30 GiB ceiling (300 publications measured 17.4 GiB)."""
    from ..recall.bigquery_patents import fetch_by_pub_nums
    pubs = [c.pub_num for c in cands if c.pub_num]
    out: dict[str, str] = {}
    calls = 0
    for i in range(0, len(pubs), CLAIMS_CHUNK):
        got = await fetch_by_pub_nums(pubs[i:i + CLAIMS_CHUNK], with_claims=True)
        calls += 1
        out.update({k: (v.get("claims_text") or "") for k, v in got.items()})
    return out, calls


def _seed_meta(docs: list[dict]) -> list[dict]:
    """Inventor / applicant of the GOOD patents, for the same-party move."""
    out = []
    for d in docs[:20]:
        raw = d.get("raw") or {}
        out.append({"inventor": raw.get("inventor") or raw.get("first_inventor"),
                    "applicant": raw.get("applicant") or raw.get("first_applicant")})
    return [m for m in out if m["inventor"] or m["applicant"]]


async def run_rounds(elements: list[dict], cpc_groups: list[str], title_terms: list[str],
                     before: str | None, round0_moves, event=None) -> dict:
    """`round0_moves()` is awaited once and returns (candidates, MoveResults)
    for the seed round — the caller supplies it because it owns the ReAct
    budget and the paper neighbourhood. Returns the loop's record."""
    pool: dict[str, Candidate] = {}
    docs_by_pub: dict[str, dict] = {}
    all_rows: list[dict] = []
    good_docs: list[dict] = []
    cover: dict[str, int] = {e["id"]: 0 for e in elements}
    stop = None
    t_start = time.monotonic()

    cands0, results0 = await round0_moves()
    for c in cands0:
        pool.setdefault((c.pub_num or c.title).upper(), c)

    round_no = 0
    while True:
        results = results0 if round_no == 0 else await _later_round(
            good_docs, elements, cpc_groups, title_terms, before, set(pool), round_no)
        for r in results:
            all_rows.append(r.row())
            for c in r.candidates:
                pool.setdefault((c.pub_num or c.title).upper(), c)
        batch = M.round_budget(results, cap=M.ROUND_CLAIMS_CAP)
        claims, bq_calls = await _claims_for(batch)
        docs = [{"pub_num": c.pub_num, "title": c.title, "sources": c.sources,
                 "abstract": c.abstract, "raw": c.raw} for c in batch]
        stats = await G.judge(elements, docs, claims)
        stats["bq_calls"] = bq_calls
        new_good = 0
        for d in docs:
            if d.get("good"):
                key = (d.get("pub_num") or "").upper()
                if key not in docs_by_pub:
                    docs_by_pub[key] = d
                    good_docs.append(d)
                    new_good += 1
        cover = G.coverage(good_docs, elements)
        if event:
            event("m1_round", f"round {round_no}: {len(batch)} claims read, {new_good} new GOOD, "
                              f"coverage {sorted(cover.values())}",
                  {"round": round_no, "read": len(batch), "new_good": new_good, "coverage": cover,
                   "moves": [r.row() for r in results], "judge": stats})
        all_rows.append({"move": "_round", "round": round_no, "read": len(batch), "new_good": new_good,
                         "judge": stats, "coverage": dict(cover)})
        stop = M.done(cover, new_good, round_no + 1)
        if stop:
            break
        round_no += 1

    return {"pool": list(pool.values()), "good": G.rank_good(good_docs), "rows": all_rows,
            "rounds": round_no + 1, "stop": stop, "coverage": cover,
            "seconds": round(time.monotonic() - t_start, 1)}


async def _later_round(good_docs: list[dict], elements: list[dict], cpc_groups: list[str],
                       title_terms: list[str], before: str | None, known: set[str],
                       round_no: int) -> list[M.MoveResult]:
    """Rounds 1+: every move starts from the GOOD set."""
    seeds = [d["pub_num"] for d in G.rank_good(good_docs)[:M.SEEDS_PER_ROUND] if d.get("pub_num")]
    uncovered = [e["id"] for e in elements if not any(e["id"] in (d.get("good_touches") or {}) for d in good_docs)]
    groups = list(dict.fromkeys(cpc_groups + _groups_of(good_docs)))[:4]
    tasks = [
        M.p1_citations(seeds, known, round_no=round_no),
        M.p1_citations(seeds, known, examiner_only=True, round_no=round_no),
        M.p4_similar(seeds, known, round_no=round_no),
        M.p3_cited_papers(seeds, known, round_no=round_no),
        M.p5_cpc_enum(groups, title_terms, before, known, round_no=round_no),
        M.p6_same_party(_seed_meta(good_docs), before, known, round_no=round_no),
    ]
    out = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[M.MoveResult] = []
    for name, r in zip(("P1_citations", "P2_examiner", "P4_similar", "P3_cited_papers", "P5_cpc_enum", "P6_same_party"), out):
        if isinstance(r, Exception):
            results.append(M.MoveResult(name=name, round=round_no, error=f"{type(r).__name__}: {r}"[:200]))
        else:
            results.append(r)
    if uncovered:
        results[-1].note += f" | uncovered: {','.join(uncovered[:6])}"
    return results


def _groups_of(good_docs: list[dict], top: int = 3) -> list[str]:
    """CPC main groups the GOOD documents actually sit in — better than the
    predicted ones once we have real hits."""
    from collections import Counter
    c: Counter = Counter()
    for d in good_docs:
        for code in ((d.get("raw") or {}).get("cpc") or []):
            g = str(code).split("/")[0].replace(" ", "")
            if len(g) >= 5:
                c[g] += 1
    return [g for g, _ in c.most_common(top)]
