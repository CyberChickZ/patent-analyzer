import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.recall import uspto_odp as odp


def test_cpc_prefix_matches_the_padded_symbol():
    # measured against the live index: H04N\ \ \ 7* -> 68,691, H04N* -> 423,653, H04N7* -> none
    assert odp._cpc_prefix("H04N7") == "H04N\\ \\ \\ 7*"
    assert odp._cpc_prefix("H04N7/15") == odp._cpc_prefix("H04N7")
    assert odp._cpc_prefix("A61B2503") == "A61B2503*"
    assert odp._cpc_prefix("H04N") == "H04N*"


def test_query_puts_title_cpc_and_the_date_cutoff_together():
    q = odp._q(["telepresence robot", "swivel"], "G05D1", "20110202")
    assert 'inventionTitle:("telepresence robot" OR swivel)' in q
    assert "cpcClassificationBag:G05D\\ \\ \\ 1*" in q
    assert "filingDate:[1900-01-01 TO 2011-02-02]" in q
    assert odp._q([], None, None) == ""


def test_enumerate_group_pages_to_exhaustion(monkeypatch):
    pages = {0: 100, 100: 100, 200: 43}
    calls = []

    async def fake_search(title_terms, cpc=None, before=None, size=100, offset=0):
        calls.append(offset)
        from patent_analyzer.recall.pool import Candidate
        return [Candidate(pub_num=f"US{offset + i}A1", match_type="Patent") for i in range(pages.get(offset, 0))], 243, None
    monkeypatch.setattr(odp, "search_patents", fake_search)
    got, total, err = asyncio.run(odp.enumerate_group(["marker"], "G05D1", "20131031"))
    assert err is None and total == 243 and len(got) == 243 and calls == [0, 100, 200]
