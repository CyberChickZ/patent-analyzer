"""Paper neighbourhood: the input paper's own citation graph on Semantic
Scholar, plus keyword searches, with OpenAlex ids so the Reliance-on-Science
bridge (patents citing these papers) can turn papers into patent seeds.

S2 Academic Graph API (swagger, api.semanticscholar.org/graph/v1):
/paper/search/match (title → paper), /paper/{id}/references and
/citations (paginated, "offset=…&limit=500"), recommendations API
/papers/forpaper/{id}. All calls go through the machine-wide 1 req/s lock.
OpenAlex ids: S2 externalIds.MAG is the OpenAlex work number (W<MAG>);
others resolved from DOI in batches of 50.
"""

from __future__ import annotations

import os
import time

from ..recall import openalex as oa
from ..recall import semantic_scholar as ss
from ..recall.pool import Candidate

MAX_REFS = int(os.environ.get("NEIGH_MAX_REFS", "1000"))
MAX_CITS = int(os.environ.get("NEIGH_MAX_CITS", "500"))
HOP2_TOP = int(os.environ.get("NEIGH_HOP2_TOP", "20"))
HOP2_REFS = int(os.environ.get("NEIGH_HOP2_REFS", "200"))
KW_PER_CAND = 2


def _key(c: Candidate) -> str:
    pid = ((c.raw or {}).get("semantic_scholar") or {}).get("paperId")
    return (pid or c.doi or c.arxiv_id or c.title[:80]).lower()


def _oa_from_s2(c: Candidate) -> str:
    ext = ((c.raw or {}).get("semantic_scholar") or {}).get("externalIds") or {}
    mag = ext.get("MAG")
    return f"W{mag}" if mag else ""


async def locate(title: str, doi: str = "", arxiv_id: str = "") -> Candidate | None:
    if doi:
        cands, _ = await ss.batch([f"DOI:{doi}"])
        if cands:
            return cands[0]
    if arxiv_id:
        cands, _ = await ss.batch([f"ARXIV:{arxiv_id}"])
        if cands:
            return cands[0]
    return await ss.match_title(title) if title else None


def _pre_cutoff(c: Candidate, cutoff: str | None) -> bool:
    if not cutoff or not c.year:
        return True
    try:
        return int(str(c.year)[:4]) <= int(cutoff[:4])
    except ValueError:
        return True


async def paper_neighbourhood(title: str, cands: list[dict], cutoff: str | None = None, doi: str = "",
                              arxiv_id: str = "", summary: str = "", embed=None) -> tuple[list[Candidate], dict]:
    """Returns (papers, info). papers carry raw['neigh'] = {source, oa_id}."""
    t0 = time.monotonic()
    info: dict = {"located": None, "sources": {}, "n": 0, "with_oa_id": 0, "s2_calls": 0}
    pool: dict[str, Candidate] = {}

    def _add(items: list[Candidate], source: str):
        n = 0
        for c in items:
            if not c.title or not _pre_cutoff(c, cutoff):
                continue
            k = _key(c)
            if k not in pool:
                c.raw.setdefault("neigh", {})["source"] = source
                pool[k] = c
                n += 1
        info["sources"][source] = info["sources"].get(source, 0) + n

    seed = await locate(title, doi, arxiv_id)
    info["s2_calls"] += 1
    if seed:
        pid = seed.raw["semantic_scholar"]["paperId"]
        info["located"] = {"paperId": pid, "title": seed.title, "year": seed.year, "doi": seed.doi}
        refs, _ = await ss.references_all(pid, MAX_REFS)
        cits, _ = await ss.citations_all(pid, MAX_CITS)
        recs, _ = await ss.recommendations(pid, limit=100)
        info["s2_calls"] += 3 + len(refs) // 500 + len(cits) // 500
        _add(refs, "references")
        _add(cits, "citations")
        _add(recs, "recommendations")
    for cand in cands[:4]:
        concept = " ".join(str(cand.get("concept") or "").split())[:300]
        if not concept:
            continue
        s2, _ = await ss.search(concept, limit=50)
        info["s2_calls"] += 1
        _add(s2, f"s2_search:{cand.get('id')}")
        oa_hits, _ = await oa.search_works(concept, limit=50)
        _add(oa_hits, f"openalex:{cand.get('id')}")
    # second hop: references of the papers closest to the invention summary
    hop_src = [c for c in pool.values() if ((c.raw or {}).get("semantic_scholar") or {}).get("paperId")]
    if hop_src and summary:
        try:
            if embed is None:
                from ..encoders import embed_docs, embed_queries
                embed = (embed_docs, embed_queries)
            import numpy as np
            dv = np.asarray(embed[0]([(c.title + " " + (c.abstract or "")[:500]) for c in hop_src]), dtype=np.float32)
            qv = np.asarray(embed[1]([summary[:2000]]), dtype=np.float32)[0]
            dv /= np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
            qv /= np.linalg.norm(qv) + 1e-9
            order = np.argsort(-(dv @ qv))[:HOP2_TOP]
            for i in order:
                pid = hop_src[int(i)].raw["semantic_scholar"]["paperId"]
                refs2, _ = await ss.references_all(pid, HOP2_REFS)
                info["s2_calls"] += 1
                _add(refs2, "hop2_references")
        except Exception as exc:
            info["hop2_error"] = f"{type(exc).__name__}: {exc}"[:200]
    # OpenAlex ids
    need = []
    for c in pool.values():
        oa_id = _oa_from_s2(c) or ((c.raw or {}).get("openalex") or {}).get("id", "").rsplit("/", 1)[-1]
        if oa_id:
            c.raw.setdefault("neigh", {})["oa_id"] = oa_id
        elif c.doi:
            need.append(c)
    if need:
        found = await oa.ids_for_dois([c.doi for c in need])
        for c in need:
            oid = found.get(c.doi.lower().replace("https://doi.org/", ""))
            if oid:
                c.raw.setdefault("neigh", {})["oa_id"] = oid
    papers = list(pool.values())
    info["n"] = len(papers)
    info["with_oa_id"] = sum(1 for c in papers if (c.raw.get("neigh") or {}).get("oa_id"))
    info["seconds"] = round(time.monotonic() - t0, 1)
    return papers, info
