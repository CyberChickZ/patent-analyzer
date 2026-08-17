import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic import neighbourhood as N
from patent_analyzer.recall.pool import Candidate


def _p(pid, title, year, mag=None, doi=""):
    return Candidate(title=title, match_type="Paper", year=str(year), doi=doi, abstract="",
                     raw={"semantic_scholar": {"paperId": pid, "externalIds": {"MAG": mag} if mag else {}}})


def test_neighbourhood_collects_sources_filters_cutoff_and_resolves_ids(monkeypatch):
    async def locate(title, doi="", arxiv_id=""):
        return _p("P0", title, 2011, mag=1516158759, doi="10.1/x")

    async def refs(pid, max_total=1000):
        return ([_p("R1", "old ref", 2005, mag=111), _p("R2", "ref with doi", 2008, doi="10.2/r2")] if pid == "P0"
                else [_p("H1", "hop two", 2001, mag=222)]), None

    async def cits(pid, max_total=1000):
        return [_p("C1", "later citation", 2015, mag=333), _p("C2", "earlier citation", 2010, mag=444)], None

    async def recs(pid, limit=100):
        return [_p("K1", "recommended", 2009, mag=555)], None

    async def s2search(q, limit=50):
        return [_p("S1", "keyword hit", 2007, mag=666)], None

    async def oasearch(q, limit=50):
        return [Candidate(title="openalex hit", match_type="Paper", year="2006", raw={"openalex": {"id": "https://openalex.org/W777"}})], None

    async def ids_for_dois(dois):
        return {"10.2/r2": "W888"}
    monkeypatch.setattr(N, "locate", locate)
    monkeypatch.setattr(N.ss, "references_all", refs)
    monkeypatch.setattr(N.ss, "citations_all", cits)
    monkeypatch.setattr(N.ss, "recommendations", recs)
    monkeypatch.setattr(N.ss, "search", s2search)
    monkeypatch.setattr(N.oa, "search_works", oasearch)
    monkeypatch.setattr(N.oa, "ids_for_dois", ids_for_dois)
    embed = (lambda texts: [[1.0, 0.0]] * len(texts), lambda texts: [[1.0, 0.0]])
    papers, info = asyncio.run(N.paper_neighbourhood("T", [{"id": "inv1", "concept": "c"}], cutoff="20110202", summary="s", embed=embed))
    titles = {c.title for c in papers}
    assert "later citation" not in titles and "earlier citation" in titles and "hop two" in titles
    assert info["sources"]["references"] == 2 and info["sources"]["citations"] == 1 and info["sources"]["hop2_references"] == 1
    ids = {c.title: (c.raw.get("neigh") or {}).get("oa_id") for c in papers}
    assert ids["old ref"] == "W111" and ids["ref with doi"] == "W888" and ids["openalex hit"] == "W777"
    assert info["with_oa_id"] == len(papers) and info["located"]["paperId"] == "P0"
