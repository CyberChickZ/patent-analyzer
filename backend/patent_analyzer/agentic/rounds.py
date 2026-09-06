"""M1 round controller: run the moves, read the claims, keep what is STRONG.

Round 0 starts from the input paper alone (predicted CPC enumeration, the
paper's own neighbourhood bridge, and the ReAct queries). Round 1 walks one
hop out from what round 0 read — from the best of it by (elements touched,
cosine), NOT only from the GOOD ones. Rounds 2+ expand only from the GOOD set.

That split is the m1a lesson. The first version expanded from GOOD at every
round and reached 0 of 5 gold families on US20120194631A1, where the old wide
loop reached 3 (h1h) and 4 (h1i). The gold in those runs came from the
one-hop citations of a query hit, and those hits were mostly NOT GOOD
themselves — a bridge document does not have to claim any element of the
invention. So "expand only from GOOD" cannot hold at the first hop; it holds
once there is a GOOD set worth walking from (leader, 2026-09-18).
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


# Sources whose candidates are read whatever the cosine says. Of the four gold families h1i
# reached on US20120194631A1, two came straight off a ReAct query, one off Lens and one off the
# citation expansion — an embedding cut that ranks them 400th of 2,140 loses three of the four
# before anything reads them (leader, 2026-09-18).
MUST_READ_SOURCES = {"lens_bridge", "lens_search", "reliance_bridge", "citation_graph",
                     "google_similar", "P2_examiner"}
MUST_READ_QUERY_RANK = int(os.environ.get("M1_MUST_READ_RANK", "10"))


def _cos_of(elements: list[dict], cands: list[Candidate], summary: str = "") -> list[float]:
    """Cosine of each candidate against the element texts, written to
    `raw["cos"]` as well so a candidate the budget did not read can still be
    ranked as a seed later."""
    rows = [{"title": c.title, "abstract": c.abstract or c.snippet, "sources": c.sources} for c in cands]
    try:
        from .prune import stage1_embed
        stage1_embed(elements, rows, topk=1, cap=1)
    except Exception:
        for c in cands:
            c.raw.setdefault("cos", 0.0)
        return [0.0] * len(cands)
    out = []
    for c, r in zip(cands, rows):
        v = float(r.get("prune_cos") or 0.0)
        c.raw["cos"] = v
        out.append(v)
    return out


def select_for_reading(elements: list[dict], cands: list[Candidate], n: int,
                       summary: str = "") -> tuple[list[Candidate], dict]:
    """Which of a round-0 channel's candidates the claims budget reads.

    m1a and m1b took the first `n` in channel order — 300 of 2,140 — and
    everything past the cut never entered the pool either, so the run could not
    even be compared with h1h/h1i, whose pool reach counted what the channels
    returned. The whole list stays in the pool now and only the READING is
    selected.

    m1c then showed that "must-read" alone does not select anything: 2,227 of
    3,229 candidates matched it (Lens alone returns hundreds), so `must[:300]`
    was again an arbitrary cut — and it dropped US20090147070A1, a gold family
    that came in through lens_search. So the must-read set is ordered too, in
    three tiers, each ranked by cosine inside itself:

      1. every query's top MUST_READ_QUERY_RANK — bounded by the query count,
         and where h1i reached two of its four gold families directly;
      2. the sources that reach gold without going through a text ranking
         (Lens, the Reliance bridge, the citation graph, Google similar,
         examiner citations);
      3. everything else.
    """
    cos = _cos_of(elements, cands, summary)
    tiers: list[list[tuple[int, Candidate]]] = [[], [], []]
    for i, c in enumerate(cands):
        rank = ((c.raw or {}).get("loop") or {}).get("rank")
        if isinstance(rank, int) and rank < MUST_READ_QUERY_RANK:
            tiers[0].append((i, c))
        elif set(c.sources or []) & MUST_READ_SOURCES:
            tiers[1].append((i, c))
        else:
            tiers[2].append((i, c))
    for t in tiers:
        t.sort(key=lambda ic: -cos[ic[0]])
    picked: list[Candidate] = []
    for t in tiers:
        picked += [c for _, c in t[:max(0, n - len(picked))]]
    read = {id(c) for c in picked}
    left = [cos[i] for i, c in enumerate(cands) if id(c) not in read]
    info = {"in": len(cands), "read": len(picked),
            "tiers": [len(t) for t in tiers],
            "read_per_tier": [sum(1 for _, c in t if id(c) in read) for t in tiers],
            "cut_cos": round(max(left), 4) if left else None}
    return picked, info


def _score_cos(elements: list[dict], docs: list[dict]) -> None:
    """Set `prune_cos` on every doc (te005 against the element texts) so the
    wide round can rank documents the judge found nothing in."""
    if not docs or not elements:
        return
    try:
        from .prune import stage1_embed
        stage1_embed(elements, docs, topk=1, cap=1)          # we only want the side effect
    except Exception:
        for d in docs:
            d.setdefault("prune_cos", 0.0)


async def _seed_meta(docs: list[dict], cap: int = 20) -> list[dict]:
    """Inventor / applicant of the seed patents, for the same-party move.

    m1a never called P6 once: this read `raw.inventor` and `raw.applicant`,
    which no BigQuery candidate has — amie_patents.pubs has no such column and
    neither does any other table we hold. The names come from USPTO ODP
    instead, one batched query for up to `cap` publication numbers. ODP's
    search projection answers firstInventorName but leaves firstApplicantName
    null, so this walks inventors only.
    """
    from ..recall import uspto_odp as odp
    pubs = [d["pub_num"] for d in docs[:cap] if d.get("pub_num")]
    if not pubs:
        return []
    try:
        got = await odp.find_many_by_publication(pubs)
    except Exception:
        return []
    out = [odp.party_names(w) for w in got.values()]
    return [m for m in out if m.get("inventor") or m.get("applicant")]


def _terms_for(elements: list[dict], ids, cap: int = 12) -> list[str]:
    """Single title words from the named elements' `thing` forms — what the
    ODP title index can match. Called with the uncovered element ids so the
    enumeration moves towards what is still missing rather than repeating the
    round-0 term set."""
    want = set(ids or ())
    pick = [e for e in elements if e.get("id") in want] or elements
    out: list[str] = []
    for e in pick:
        for form in ((e.get("facets") or {}).get("thing") or [])[:3]:
            for w in str(form).lower().split():
                if len(w) > 3 and w not in out:
                    out.append(w)
    return out[:cap]


async def run_rounds(elements: list[dict], cpc_groups: list[str], title_terms: list[str],
                     before: str | None, round0_moves, event=None,
                     offsets: dict[str, int] | None = None) -> dict:
    """`round0_moves()` is awaited once and returns (candidates, MoveResults)
    for the seed round — the caller supplies it because it owns the ReAct
    budget and the paper neighbourhood. Returns the loop's record."""
    pool: dict[str, Candidate] = {}
    docs_by_pub: dict[str, dict] = {}
    all_rows: list[dict] = []
    good_docs: list[dict] = []
    judged: list[dict] = []
    offsets = {} if offsets is None else offsets  # P5's paging cursor, shared with round 0
    cover: dict[str, int] = {e["id"]: 0 for e in elements}
    uncovered = [e["id"] for e in elements]
    stop = None
    dry = 0
    t_start = time.monotonic()

    cands0, results0 = await round0_moves()
    for c in cands0:
        pool.setdefault((c.pub_num or c.title).upper(), c)

    round_no = 0
    while True:
        if round_no == 0:
            results = results0
        elif round_no == 1:
            results = await _wide_round(judged, list(pool.values()), before, set(pool), round_no)
        else:
            results = await _later_round(good_docs, elements, cpc_groups, title_terms, before,
                                         set(pool), round_no, uncovered, offsets)
        for r in results:
            all_rows.append(r.row())
            for c in r.candidates:
                pool.setdefault((c.pub_num or c.title).upper(), c)
        batch = M.round_budget(results, cap=M.ROUND0_CLAIMS_CAP if round_no == 0 else M.ROUND_CLAIMS_CAP)
        claims, bq_calls = await _claims_for(batch)
        docs = [{"pub_num": c.pub_num, "title": c.title, "sources": c.sources,
                 "abstract": c.abstract, "raw": c.raw} for c in batch]
        _score_cos(elements, docs)
        stats = await G.judge(elements, docs, claims)
        stats["bq_calls"] = bq_calls
        judged.extend(docs)
        new_good = new_strong = 0
        for d in docs:
            if d.get("good"):
                key = (d.get("pub_num") or "").upper()
                if key not in docs_by_pub:
                    docs_by_pub[key] = d
                    good_docs.append(d)
                    new_good += 1
                    new_strong += 1 if G.is_strong(d) else 0
        cover = G.coverage(good_docs, elements)
        uncovered = G.uncovered(cover)
        dry = 0 if new_strong else dry + 1
        row = {"move": "_round", "round": round_no, "read": len(batch), "new_good": new_good,
               "new_strong": new_strong, "judge": stats, "coverage": dict(cover),
               "uncovered": list(uncovered), "dry_streak": dry}
        if event:
            event("m1_round", f"round {round_no}: {len(batch)} claims read, {new_good} new GOOD "
                              f"({new_strong} strong), {len(uncovered)} elements still uncovered", row)
        all_rows.append(row)
        stop = M.done(cover, new_strong, round_no + 1, dry)
        if stop:
            break
        round_no += 1

    return {"pool": list(pool.values()), "good": G.rank_good(good_docs), "rows": all_rows,
            "rounds": round_no + 1, "stop": stop, "coverage": cover, "uncovered": uncovered,
            "strong": sum(1 for d in good_docs if G.is_strong(d)),
            # every publication whose claims were actually read — the pool is what the channels
            # found, this is what the budget could afford to look at, and the two reaches differ
            "read": [d["pub_num"] for d in judged if d.get("pub_num")],
            "seconds": round(time.monotonic() - t_start, 1)}


async def _wide_round(judged: list[dict], pool: list[Candidate], before: str | None,
                      known: set[str], round_no: int) -> list[M.MoveResult]:
    """Round 1: one hop out from the best of round 0, GOOD or not.

    A document that cites the gold does not have to claim any element itself,
    so requiring GOOD here throws away the bridges. Ranking is (elements
    touched, cosine), and it runs over the WHOLE round-0 pool rather than only
    over what the claims budget could afford to read — a high-cosine query hit
    that did not make the reading cut is still a perfectly good bridge to walk
    from, and seeding costs nothing per seed."""
    ranked = sorted(judged, key=lambda d: (-len(d.get("good_touches") or {}),
                                           -float(d.get("prune_cos") or 0.0)))
    seeds = [d["pub_num"] for d in ranked if d.get("pub_num")]
    seen = set(seeds)
    unread = sorted((c for c in pool if c.pub_num and c.pub_num not in seen),
                    key=lambda c: -float((c.raw or {}).get("cos") or 0.0))
    seeds = (seeds + [c.pub_num for c in unread])[:M.WIDE_SEEDS]
    out = await asyncio.gather(
        M.p1_citations(seeds, known, cap=M.CAPS["W1_citations"], round_no=round_no),
        M.p4_similar(seeds, known, cap=M.CAPS["W4_similar"], round_no=round_no),
        return_exceptions=True)
    results: list[M.MoveResult] = []
    for name, r in zip(("W1_citations", "W4_similar"), out):
        if isinstance(r, Exception):
            results.append(M.MoveResult(name=name, round=round_no, error=f"{type(r).__name__}: {r}"[:200]))
        else:
            r.name = name
            r.note = (r.note + f" | {len(seeds)} seeds, GOOD not required").strip(" |")
            results.append(r)
    return results


async def _later_round(good_docs: list[dict], elements: list[dict], cpc_groups: list[str],
                       title_terms: list[str], before: str | None, known: set[str],
                       round_no: int, uncovered: list[str],
                       offsets: dict[str, int]) -> list[M.MoveResult]:
    """Rounds 2+: every move starts from the GOOD set, aimed at what is still
    uncovered — the GOOD documents that touch an uncovered element seed first,
    their CPC groups are enumerated first, and the ODP title terms come from
    those elements' own `thing` forms."""
    aimed = G.rank_for(good_docs, uncovered, M.SEEDS_PER_ROUND)
    seeds = [d["pub_num"] for d in aimed if d.get("pub_num")]
    touching = [d for d in good_docs if set(uncovered) & set(d.get("good_touches") or {})]
    groups = list(dict.fromkeys(_groups_of(touching) + _groups_of(good_docs) + cpc_groups))[:4]
    terms = _terms_for(elements, uncovered) or title_terms
    meta = await _seed_meta(aimed)
    tasks = [
        M.p1_citations(seeds, known, round_no=round_no),
        M.p1_citations(seeds, known, examiner_only=True, round_no=round_no),
        M.p4_similar(seeds, known, round_no=round_no),
        M.p3_cited_papers(seeds, known, round_no=round_no),
        M.p5_cpc_enum(groups, terms, before, known, round_no=round_no, offsets=offsets),
        M.p6_same_party(meta, before, known, round_no=round_no),
    ]
    out = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[M.MoveResult] = []
    for name, r in zip(("P1_citations", "P2_examiner", "P4_similar", "P3_cited_papers", "P5_cpc_enum", "P6_same_party"), out):
        if isinstance(r, Exception):
            results.append(M.MoveResult(name=name, round=round_no, error=f"{type(r).__name__}: {r}"[:200]))
        else:
            results.append(r)
    aim = f"aimed at {len(uncovered)} uncovered: {','.join(uncovered[:6])}"
    for r in results:
        r.note = (r.note + f" | {aim}").strip(" |")
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
