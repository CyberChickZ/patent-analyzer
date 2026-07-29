import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.quote_verify import locate_quote, verify_checklist_results

DOC = ("[0001] The apparatus comprises a virtual channel unit configured to time-share "
       "a serial bus between a first virtual channel and a second virtual channel.\n"
       "[0002] A buffering control unit receives data via the first virtual channel; "
       "the switching unit stores that data in the second receive buffer when the "
       "link partner lacks VC1 support.")


def test_exact_quote_found():
    found, sim = locate_quote("a buffering control unit receives data via the first virtual channel", DOC)
    assert found and sim == 1.0


def test_punctuation_and_case_insensitive():
    found, _ = locate_quote("Time-Share a Serial Bus, between a first virtual channel", DOC)
    assert found


def test_minor_ocr_noise_still_found():
    found, sim = locate_quote("the switching unit stores that data in the secnod receive buffer when", DOC)
    assert found and sim >= 0.9


def test_fabricated_quote_rejected():
    found, sim = locate_quote("the apparatus uses quantum tunneling to encrypt every packet", DOC)
    assert not found and sim < 0.9


def test_short_quote_rejected():
    assert locate_quote("serial", DOC) == (False, 0.0)


def test_verify_downgrades_unverifiable():
    cr = {
        "vc unit": {"score": 2, "evidence_quote": "virtual channel unit configured to time-share a serial bus"},
        "encrypt": {"score": 2, "evidence_quote": "packets are encrypted with a rotating key schedule"},
        "no quote": {"score": 1, "evidence_quote": ""},
        "absent": {"score": 0},
    }
    stats = verify_checklist_results(cr, DOC)
    assert cr["vc unit"]["score"] == 2
    assert cr["encrypt"]["score"] == 0 and cr["encrypt"]["quote_unverified"]
    assert cr["no quote"]["score"] == 0 and cr["no quote"]["match"] is False
    assert cr["absent"]["score"] == 0 and "quote_unverified" not in cr["absent"]
    assert stats == {"scored": 3, "with_quote": 2, "quotes": 2, "verified": 1, "downgraded": 2}


def test_verify_keeps_multiple_quotes_and_downgrades_only_when_none_survive():
    cr = {"buffer": {"score": 2, "evidence_quotes": [
        "a buffering control unit receives data via the first virtual channel",
        "the switching unit stores that data in the second receive buffer",
        "packets are encrypted with a rotating key schedule"]}}
    stats = verify_checklist_results(cr, DOC)
    item = cr["buffer"]
    assert item["score"] == 2 and len(item["verified_quotes"]) == 2
    assert [c["verified"] for c in item["quote_checks"]] == [True, True, False]
    assert item["evidence_quote"] == item["verified_quotes"][0]
    assert stats["quotes"] == 3 and stats["verified"] == 2 and stats["downgraded"] == 0
