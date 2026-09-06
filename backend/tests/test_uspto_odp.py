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


def test_pub_digits_does_not_swallow_the_kind_code_digit():
    """N4 probe, 2026-09-18: stripping every non-digit made "US10963506B2"
    into "109635062", which matches nothing — so the file wrapper, the SRNT
    search record, the 892/1449 citation lists and the drawings were
    unreachable for every pooled number carrying a B2 / B1 / A1 suffix."""
    from patent_analyzer.recall.uspto_odp import pub_digits
    assert pub_digits("US10963506B2") == "10963506"
    assert pub_digits("US10963506") == "10963506"
    assert pub_digits("10963506") == "10963506"
    assert pub_digits("US-8,395,653-B2") == "8395653"
    assert pub_digits("US20120194631A1") == "20120194631"     # pre-grant: year + 7, kind dropped
    assert pub_digits("US6606111B1") == "6606111"
    assert pub_digits("") == ""
