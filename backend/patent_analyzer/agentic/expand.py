"""Second hop from seed patents via our own BigQuery tables: examiner
(SEA) citations become new candidates; CPC subclass of the seeds feeds
the next query round. Point lookups only."""

from __future__ import annotations

import re
from collections import Counter

from ..recall.bigquery_patents import fetch_by_pub_nums, fetch_citations, fetch_cited_by, fetch_meta_light
from ..recall.pool import Candidate

MAX_CITED_PER_ROUND = 200
MAX_CITED_LIGHT = int(__import__("os").environ.get("EXPAND_MAX_CITED", "5000"))
FULL_META_HEAD = 200   # abstracts for the most-cited head; titles only beyond


def _canon(p: str) -> str:
    return re.sub(r"[\s\-/,.]", "", (p or "").upper())


def _after(meta: dict, cutoff: str | None) -> bool:
    """priority_date on/after the cutoff (YYYYMMDD): not prior art, never a seed."""
    d = str(meta.get("priority_date") or "").replace("-", "")[:8]
    return bool(cutoff and d and d >= cutoff)


PER_SEED_MIN = int(__import__("os").environ.get("EXPAND_PER_SEED_MIN", "3"))
PER_SEED_MAX = int(__import__("os").environ.get("EXPAND_PER_SEED_MAX", "50"))
# The budget has to grow with the seed set or a per-seed split is theatre: 5,000 slots shared by
# 43,762 seeds is one document each for the first 5,000 and nothing for the rest (leader/Harry,
# 2026-09-19). 5 per seed, floored at the old 5,000 so small jobs are unchanged and capped at
# 20,000 so the metadata fetch stays inside its 15 GiB guard.
BUDGET_PER_SEED = int(__import__("os").environ.get("EXPAND_BUDGET_PER_SEED", "5"))
BUDGET_MAX = int(__import__("os").environ.get("EXPAND_BUDGET_MAX", "20000"))
# A query's own hit is a better place to walk from than the eight-thousandth bridge patent, so it
# gets a deeper slice of its own references.
K_BY_KIND = {"query": 10, "bridge": 3, "lens": 3}


def budget_for(n_seeds: int, floor: int) -> int:
    return max(floor, min(BUDGET_MAX, BUDGET_PER_SEED * max(1, n_seeds)))


def _per_seed(by_seed: dict[str, list[str]], cited: "Counter", known: set[str],
              meta: dict, max_cited: int, order: list[str] | None = None,
              kind: dict[str, str] | None = None) -> tuple[list[str], dict]:
    """Give every seed a share of the expansion instead of one global vote.

    The old rule was `cited.most_common()[:max_cited]` — the documents cited by
    the most seeds, capped at 5,000. That constant was chosen when a job had
    about 1,400 seeds. Every later recall improvement (named-entity queries for
    the paper channel, dates pushed into the API, the Reliance bridge entering
    the pool, the Lens bridge at its 800 cap) raised it to about 10,000, and on
    one paper 43,762 — so the same 5,000 slots were being filled from a pool of
    up to 251,000 by "how many seeds cite this", which at that size selects the
    field's textbook references rather than this invention's. Measured over 8
    papers: the citation expansion first reached 19 gold families before and 3
    after, while every other channel held (bridges 11 = 11). H.md §H1.10.

    So each seed takes its own top k, k = ceil(max_cited / seeds) clamped to
    [PER_SEED_MIN, PER_SEED_MAX]. Inside a seed the order is still the global
    weight, which keeps the "corroborated by other seeds" signal as a tie-break
    where it belongs — between one seed's own references — instead of letting
    it decide the whole expansion. If the union still overflows, it is trimmed
    round-robin so a seed is never dropped whole.

    With more seeds than slots — 10,000 seeds against 5,000 — not every seed can
    have even one, so the order they are served in decides. `order` is the
    caller's own seed order, and loop.py builds it query hits first, then the
    Reliance bridge, then Lens; a query's own hit is a better place to walk from
    than the eight-thousandth bridge patent.

    Not implemented, and it cannot be here: Harry asked for the overflow tie to
    go to the seed with the most claims-judged element touches. Expansion runs
    during recall and the claims judge runs after it, so no such number exists
    yet at this point. The caller's ordering is the best signal actually
    available, and it is at least the one the caller controls.

    Worth knowing before reading `per_seed_k`: when seeds outnumber slots the
    allocation degenerates to one document per seed for the first `max_cited`
    seeds, and the rest get nothing however large k says it is. wg4 had 3,513
    to 43,762 seeds against 5,000 slots, so that is the normal case today, not
    an edge. Making this better needs either a cap that scales with the seed
    count or a smaller, better seed set — a per-seed split alone cannot conjure
    slots that do not exist. `per_seed_served` reports how many seeds actually
    got one.
    """
    rank = {s: i for i, s in enumerate(order or [])}
    seeds = sorted((s for s, ps in by_seed.items() if ps),
                   key=lambda s: rank.get(s, len(rank)))
    if not seeds:
        return [], {"per_seed_k": 0, "per_seed_seeds": 0}
    k = max(PER_SEED_MIN, min(PER_SEED_MAX, -(-max_cited // len(seeds))))
    kind = kind or {}
    queues: list[list[str]] = []
    for s in seeds:
        fresh = [p for p in dict.fromkeys(by_seed[s]) if p not in known and p not in meta]
        fresh.sort(key=lambda p: -cited[p])
        queues.append(fresh[:max(k, K_BY_KIND.get(kind.get(s, ""), 0))])
    out, seen, served = [], set(), set()
    for i in range(max([k] + [len(q) for q in queues] or [k])):   # round robin: no seed dropped whole
        for s, q in zip(seeds, queues):
            if i < len(q) and q[i] not in seen and len(out) < max_cited:
                seen.add(q[i])
                out.append(q[i])
                served.add(s)                            # this seed actually contributed one
        if len(out) >= max_cited:
            break
    return out, {"per_seed_k": k, "per_seed_seeds": len(seeds), "per_seed_served": len(served)}


async def expand(seed_pubs: list[str], known: set[str], max_cited: int = MAX_CITED_PER_ROUND,
                 before: str | None = None, light: bool = False, forward: bool = False,
                 seed_kind: dict[str, str] | None = None) -> tuple[list[Candidate], dict]:
    """Returns (new candidates from citations, info) where info has the
    seeds' family ids, CPC subclass counts and BQ stats. Seeds and cited
    documents with priority_date >= `before` (YYYYMMDD) are dropped.
    `light`: the head (FULL_META_HEAD most-cited) gets title+abstract, the
    rest title/family/date only via the narrow-column lookup — h1b lost 3
    gold families at citation ranks 1414-1463 to the 200 cap."""
    seeds = [s for s in dict.fromkeys(_canon(p) for p in seed_pubs if p) if s]
    info = {"seeds": len(seeds), "cpc_subclasses": {}, "families": {}, "cited_total": 0, "cited_new": 0,
            "seeds_after_cutoff": []}
    if not seeds:
        return [], info
    # light: hundreds of seeds only need family/date (16 GiB vs 4 GiB measured on 501 seeds)
    meta = await (fetch_meta_light(seeds) if light else fetch_by_pub_nums(seeds, with_claims=False))
    info["seeds_after_cutoff"] = sorted(k for k, m in meta.items() if _after(m, before))
    meta = {k: m for k, m in meta.items() if k not in info["seeds_after_cutoff"]}
    seeds = [s for s in seeds if s not in info["seeds_after_cutoff"]]
    cpc = Counter()
    for k, m in meta.items():
        info["families"][k] = m.get("family_id", "")
        for c in m.get("cpc_codes") or []:
            cpc[c[:4]] += 1
    info["cpc_subclasses"] = dict(cpc.most_common(5))

    cits = await fetch_citations(seeds)
    cited = Counter()
    by_seed: dict[str, list[str]] = {}
    for s, c in cits.items():
        for x in c.get("cits", []):
            pub = _canon(x.get("cited", ""))
            if not pub or x.get("npl_text"):
                continue
            weight = 2 if "SEA" in (x.get("category") or "") else 1
            cited[pub] += weight
            by_seed.setdefault(_canon(s), []).append(pub)
    if forward:
        # forward edges (patents citing the seed) at weight 1 — pilot's
        # citing:1 vs cited:3; date filter below drops post-cutoff ones
        try:
            fwd = await fetch_cited_by(seeds)
        except Exception as exc:          # budget guard or BQ error: forward edges are optional
            info["forward_error"] = f"{type(exc).__name__}: {exc}"[:160]
            fwd = {}
        for s, rows in fwd.items():
            for x in rows:
                pub = _canon(x.get("publication_number", ""))
                if pub:
                    cited[pub] += 1
                    by_seed.setdefault(_canon(s), []).append(pub)
        info["forward_total"] = sum(len(v) for v in fwd.values())
    info["cited_total"] = len(cited)
    new, alloc = _per_seed(by_seed, cited, known, meta, max_cited, order=seeds, kind=seed_kind)
    info["cited_new"] = len(new)
    info.update(alloc)
    kept = set(new)
    # which seed brought which kept document (ids only; the funnel joins them to gold)
    info["cited_by_seed"] = {s: [p for p in dict.fromkeys(ps) if p in kept] for s, ps in by_seed.items()}
    info["cited_by_seed"] = {s: ps for s, ps in info["cited_by_seed"].items() if ps}
    if not new:
        return [], info
    if light and len(new) > FULL_META_HEAD:
        cmeta = await fetch_by_pub_nums(new[:FULL_META_HEAD], with_claims=False)
        tail = await fetch_meta_light(new[FULL_META_HEAD:])
        for pub, m in tail.items():
            cmeta.setdefault(pub, m)
        info["cited_light"] = len(tail)
    else:
        cmeta = await fetch_by_pub_nums(new, with_claims=False)
    out = []
    for pub, m in cmeta.items():
        if _after(m, before):
            continue
        out.append(Candidate(
            title=m.get("title") or pub, snippet=(m.get("abstract") or "")[:500], abstract=m.get("abstract") or "",
            match_type="Patent", pub_num=pub, year=str(m.get("priority_date") or "")[:4],
            url=f"https://patents.google.com/patent/{pub}/en", source_score=float(cited[pub]),
            sources=["citation_graph"],
            raw={"bigquery": {"family_id": m.get("family_id", ""), "priority_date": m.get("priority_date", ""),
                              "cpc_codes": m.get("cpc_codes") or [], "cited_by_seeds": cited[pub]}},
        ))
    return out, info


MAX_SIMILAR = int(__import__("os").environ.get("EXPAND_MAX_SIMILAR", "1000"))


async def similar_neighbours(seed_pubs: list[str], known: set[str], before: str | None = None,
                             max_out: int = MAX_SIMILAR, max_seeds: int = 300) -> tuple[list[Candidate], dict]:
    """Google's published nearest neighbours (embedding_v1 `similar`) of the
    seeds, ranked by how many seeds share them, date-filtered, titles only."""
    from ..recall.bigquery_patents import fetch_similar
    seeds = [s for s in dict.fromkeys(_canon(p) for p in seed_pubs if p) if s][:max_seeds]
    info = {"seeds": len(seeds), "similar_total": 0, "similar_new": 0, "by_seed": {}}
    if not seeds:
        return [], info
    try:
        sim = await fetch_similar(seeds)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return [], info
    counts = Counter()
    for s, ns in sim.items():
        for n in ns:
            if n and n != s:
                counts[n] += 1
    info["similar_total"] = len(counts)
    new = [p for p, _ in counts.most_common() if p not in known and p not in set(seeds)][:max_out]
    info["similar_new"] = len(new)
    if not new:
        return [], info
    meta = await fetch_meta_light(new)
    kept = set()
    out = []
    for pub, m in meta.items():
        if _after(m, before):
            continue
        kept.add(pub)
        out.append(Candidate(
            title=m.get("title") or pub, snippet="", abstract="", match_type="Patent", pub_num=pub,
            year=str(m.get("priority_date") or "")[:4], url=f"https://patents.google.com/patent/{pub}/en",
            source_score=float(counts[pub]), sources=["google_similar"],
            raw={"bigquery": {"family_id": m.get("family_id", ""), "priority_date": m.get("priority_date", ""),
                              "shared_by_seeds": counts[pub]}}))
    info["by_seed"] = {s: [n for n in ns if n in kept] for s, ns in sim.items()}
    info["by_seed"] = {s: ns for s, ns in info["by_seed"].items() if ns}
    return out, info
