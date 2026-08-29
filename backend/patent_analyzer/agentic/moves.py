"""M1 recall loop: find a few of the right ones, then walk out from those.

The 8-paper funnel says where the gold actually comes from: of the 30 families
a search reached, the citation expansion first reached 19, the paper->patent
bridges 8, Lens 2, Google's similar neighbours 1 — and 60 template keyword
queries reached none (H.md §H1.5). So a query's job is to find a seed, and the
moves below do the reaching.

The difference from the old wide loop is what counts as a seed. That loop
expanded from every query hit (2,000+ publications, almost all noise). Here a
document becomes a seed only after its CLAIMS are read and the model can name
the claim number that touches an element (agentic/good.py) — and every move
brings back a bounded number per round, so the claims budget is the sum of the
caps rather than a share of a pool that has to be ranked first.

Each move records what it spent, what it brought back, how many of those were
judged GOOD, and (in evals) which gold families it reached first; a move that
brings back no GOOD over the 8-paper set gets deleted.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from ..recall.pool import Candidate

# per-round caps (leader, 2026-09-18: <=600 claims read per round, ~$1.6/job)
CAPS = {
    "P1_citations": int(os.environ.get("M1_P1", "150")),
    "P2_examiner": int(os.environ.get("M1_P2", "80")),
    "P3_cited_papers": int(os.environ.get("M1_P3", "60")),
    "P4_similar": int(os.environ.get("M1_P4", "100")),
    "P5_cpc_enum": int(os.environ.get("M1_P5", "200")),
    "P6_same_party": int(os.environ.get("M1_P6", "60")),
    "A1_paper_bridge": int(os.environ.get("M1_A1", "80")),
    "A2_author_patents": int(os.environ.get("M1_A2", "50")),
    "A3_paper_graph": int(os.environ.get("M1_A3", "80")),
    "S1_input_authors": int(os.environ.get("M1_S1", "50")),
    "S2_predicted_cpc": int(os.environ.get("M1_S2", "300")),
    "S3_react": int(os.environ.get("M1_S3", "300")),
}
ROUND_CLAIMS_CAP = int(os.environ.get("M1_ROUND_CLAIMS", "600"))
SEEDS_PER_ROUND = int(os.environ.get("M1_SEEDS", "50"))
MAX_ROUNDS = int(os.environ.get("M1_ROUNDS", "4"))
COVER_TARGET = int(os.environ.get("M1_COVER", "3"))       # each element covered by >=3 GOOD


@dataclass
class MoveResult:
    name: str
    round: int
    candidates: list[Candidate] = field(default_factory=list)
    calls: int = 0
    gib: float = 0.0
    seconds: float = 0.0
    error: str | None = None
    note: str = ""

    def row(self) -> dict:
        return {"move": self.name, "round": self.round, "brought": len(self.candidates),
                "calls": self.calls, "gib": round(self.gib, 2), "seconds": round(self.seconds, 1),
                "error": self.error, "note": self.note[:200]}


def _cands(pubs: list[str], meta: dict, source: str, cap: int) -> list[Candidate]:
    out = []
    for p in pubs[:cap]:
        m = meta.get(p) or {}
        out.append(Candidate(title=(m.get("title") or "")[:300], abstract=(m.get("abstract") or "")[:2000],
                             pub_num=p, match_type="Patent", year=(m.get("publication_date") or "")[:4],
                             sources=[source], raw={"move": source, "cpc": (m.get("cpc_codes") or [])[:8],
                                                    "family_id": m.get("family_id")}))
    return out


async def p1_citations(seeds: list[str], known: set[str], cap: int = 0, examiner_only: bool = False,
                       round_no: int = 0) -> MoveResult:
    """Backward + forward citations of the GOOD patents. `examiner_only` keeps
    the rows the search report / examiner added (category contains SEA), which
    is the same kind of edge the gold itself is."""
    from ..recall.bigquery_patents import fetch_by_pub_nums, fetch_cited_by, fetch_citations
    name = "P2_examiner" if examiner_only else "P1_citations"
    cap = cap or CAPS[name]
    r = MoveResult(name=name, round=round_no)
    if not seeds:
        return r
    t0 = time.monotonic()
    try:
        cits = await fetch_citations(seeds)
        r.calls += 1
        weight: dict[str, int] = {}
        for s, c in cits.items():
            for x in c.get("cits", []):
                pub, cat = x.get("cited", ""), (x.get("category") or "")
                if not pub or x.get("npl_text"):
                    continue
                if examiner_only and "SEA" not in cat:
                    continue
                weight[pub] = weight.get(pub, 0) + (2 if "SEA" in cat else 1)
        if not examiner_only:
            fwd = await fetch_cited_by(seeds)
            r.calls += 1
            for _s, rows in fwd.items():
                for x in rows:
                    pub = x.get("publication_number", "")
                    if pub:
                        weight[pub] = weight.get(pub, 0) + 1
        new = [p for p, _ in sorted(weight.items(), key=lambda kv: -kv[1]) if p not in known][:cap]
        meta = await fetch_by_pub_nums(new, with_claims=False) if new else {}
        r.calls += 1
        r.candidates = _cands(new, meta, name, cap)
    except Exception as exc:
        r.error = f"{type(exc).__name__}: {exc}"[:200]
    r.seconds = time.monotonic() - t0
    return r


async def p4_similar(seeds: list[str], known: set[str], cap: int = 0, round_no: int = 0) -> MoveResult:
    """Google's own embedding neighbours of the GOOD patents (amie_patents.similar)."""
    from ..recall.bigquery_patents import fetch_by_pub_nums, fetch_similar
    r = MoveResult(name="P4_similar", round=round_no)
    cap = cap or CAPS["P4_similar"]
    if not seeds:
        return r
    t0 = time.monotonic()
    try:
        sim = await fetch_similar(seeds)
        r.calls += 1
        seen: dict[str, int] = {}
        for _s, rows in sim.items():
            for x in rows:
                pub = x if isinstance(x, str) else x.get("publication_number", "")
                if pub:
                    seen[pub] = seen.get(pub, 0) + 1          # shared by several seeds ranks higher
        new = [p for p, _ in sorted(seen.items(), key=lambda kv: -kv[1]) if p not in known][:cap]
        meta = await fetch_by_pub_nums(new, with_claims=False) if new else {}
        r.calls += 1
        r.candidates = _cands(new, meta, "P4_similar", cap)
    except Exception as exc:
        r.error = f"{type(exc).__name__}: {exc}"[:200]
    r.seconds = time.monotonic() - t0
    return r


async def p5_cpc_enum(groups: list[str], title_terms: list[str], before: str | None, known: set[str],
                      cap: int = 0, round_no: int = 0, name: str = "P5_cpc_enum") -> MoveResult:
    """Enumerate a CPC main group through USPTO ODP. Google answers any query
    with its top 100; ODP pages the whole group, which is how H1-02's
    US20100010703A1 is reachable at all (H.md §H7.5)."""
    from ..recall import uspto_odp as odp
    r = MoveResult(name=name, round=round_no)
    cap = cap or CAPS.get(name, 200)
    if not groups or not title_terms:
        return r
    t0 = time.monotonic()
    per_group = max(1, cap // max(1, len(groups)))
    for g in groups:
        try:
            got, total, err = await odp.enumerate_group(title_terms, g, before=before, max_records=per_group)
        except Exception as exc:
            r.error = f"{type(exc).__name__}: {exc}"[:200]
            break
        r.calls += max(1, (len(got) + 99) // 100)
        if err:
            r.error = err
            continue
        r.note += f"{g}:{total} "
        r.candidates.extend(c for c in got if c.pub_num not in known)
        if len(r.candidates) >= cap:
            break
    r.candidates = r.candidates[:cap]
    r.seconds = time.monotonic() - t0
    return r


async def p6_same_party(seeds_meta: list[dict], before: str | None, known: set[str], cap: int = 0,
                        round_no: int = 0, name: str = "P6_same_party") -> MoveResult:
    """Other patents of the GOOD patents' inventors and applicants. ODP fields
    verified 2026-09-18: firstInventorName -> 158, inventorBag.inventorNameText
    -> 264, firstApplicantName -> 89 for one name."""
    from ..recall import uspto_odp as odp
    r = MoveResult(name=name, round=round_no)
    cap = cap or CAPS.get(name, 60)
    names: list[tuple[str, str]] = []
    for m in seeds_meta:
        for f, v in (("applicationMetaData.firstInventorName", m.get("inventor")),
                     ("applicationMetaData.firstApplicantName", m.get("applicant"))):
            if v and (f, v) not in names:
                names.append((f, v))
    if not names:
        return r
    t0 = time.monotonic()
    for field_name, value in names[:6]:
        q = f'{field_name}:"{value}"'
        if before and len(before) == 8:
            q += f" AND applicationMetaData.filingDate:[1900-01-01 TO {before[:4]}-{before[4:6]}-{before[6:]}]"
        data, err = await odp._get("/patent/applications/search", {"q": q, "limit": 100})
        r.calls += 1
        if err or not isinstance(data, dict):
            r.error = r.error or err
            continue
        for w in data.get("patentFileWrapperDataBag") or []:
            c = odp._candidate(w)
            if c and c.pub_num not in known:
                c.sources = [name]
                r.candidates.append(c)
        if len(r.candidates) >= cap:
            break
    r.candidates = r.candidates[:cap]
    r.note = "; ".join(f"{f.split('.')[-1]}={v}" for f, v in names[:6])
    r.seconds = time.monotonic() - t0
    return r


async def a1_paper_bridge(oa_ids: list[str], known: set[str], cap: int = 0, round_no: int = 0) -> MoveResult:
    """Papers -> the patents citing them (Reliance on Science, amie_patents.pcs_oa)."""
    from ..recall.bigquery_patents import fetch_by_pub_nums, fetch_citing_patents
    r = MoveResult(name="A1_paper_bridge", round=round_no)
    cap = cap or CAPS["A1_paper_bridge"]
    if not oa_ids:
        return r
    t0 = time.monotonic()
    try:
        citing = await fetch_citing_patents(oa_ids)
        r.calls += 1
        pubs: list[str] = []
        for _oid, rows in citing.items():
            pubs += [x.get("patent_pub") or "" for x in rows]
        new = [p for p in dict.fromkeys(p for p in pubs if p) if p not in known][:cap]
        meta = await fetch_by_pub_nums(new, with_claims=False) if new else {}
        r.calls += 1
        r.candidates = _cands(new, meta, "A1_paper_bridge", cap)
    except Exception as exc:
        r.error = f"{type(exc).__name__}: {exc}"[:200]
    r.seconds = time.monotonic() - t0
    return r


async def p3_cited_papers(seeds: list[str], known: set[str], cap: int = 0, round_no: int = 0) -> MoveResult:
    """GOOD patents -> the papers they cite -> back to the patents citing those
    papers (the reverse bridge, amie_patents.pcs_oa_by_patent)."""
    from ..recall.bigquery_patents import fetch_cited_papers
    r = MoveResult(name="P3_cited_papers", round=round_no)
    cap = cap or CAPS["P3_cited_papers"]
    if not seeds:
        return r
    t0 = time.monotonic()
    try:
        cited = await fetch_cited_papers(seeds)
        r.calls += 1
        oa = [x.get("oa_id") for rows in cited.values() for x in rows]
        oa = [o for o in dict.fromkeys(o for o in oa if o)][:200]
        r.note = f"{len(oa)} papers"
        if oa:
            inner = await a1_paper_bridge(oa, known, cap=cap, round_no=round_no)
            r.calls += inner.calls
            r.error = inner.error
            for c in inner.candidates:
                c.sources = ["P3_cited_papers"]
            r.candidates = inner.candidates
    except Exception as exc:
        r.error = f"{type(exc).__name__}: {exc}"[:200]
    r.seconds = time.monotonic() - t0
    return r


def round_budget(results: list[MoveResult], cap: int = ROUND_CLAIMS_CAP) -> list[Candidate]:
    """What this round sends to the claims judge: every move's candidates,
    de-duplicated, trimmed to the round's claims cap by taking from the moves
    in turn so that no single move can crowd the others out."""
    seen: set[str] = set()
    queues = [list(r.candidates) for r in results if r.candidates]
    out: list[Candidate] = []
    while queues and len(out) < cap:
        for q in list(queues):
            if not q:
                queues.remove(q)
                continue
            c = q.pop(0)
            key = (c.pub_num or c.title).upper()
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= cap:
                break
    return out


def done(cover: dict[str, int], new_good: int, round_no: int) -> str | None:
    """Why the loop stops, or None to keep going (leader 2026-09-18: every
    element covered by >=3 GOOD, or a round that adds none, or 4 rounds)."""
    if round_no >= MAX_ROUNDS:
        return f"{MAX_ROUNDS} rounds"
    if cover and all(n >= COVER_TARGET for n in cover.values()):
        return f"every element covered by >={COVER_TARGET} GOOD"
    if round_no > 0 and new_good == 0:
        return "a round added no GOOD"
    return None
