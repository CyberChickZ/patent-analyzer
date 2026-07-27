import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from extraction_errors import snap_quote

DOC = ("[0001] The apparatus comprises a virtual channel unit configured to time-share "
       "a serial bus between a first virtual channel and a second virtual channel.\n"
       "[0002] A buffering control unit receives data via the first virtual channel; "
       "the switching unit stores that data in the second receive buffer when the "
       "link partner lacks VC1 support.\n"
       "[0003] Wireless power transfer coils are arranged in a hexagonal lattice.")


def test_snap_exact_returns_original_char_span():
    q = "a buffering control unit receives data via the first virtual channel"
    found, loc = snap_quote(q, DOC)
    assert found and loc["method"] == "exact"
    s, e = loc["char"]
    assert DOC[s:e].lower() == q


def test_snap_exact_survives_punctuation_and_case():
    found, loc = snap_quote("Virtual Channel Unit, configured to TIME-SHARE a serial bus", DOC)
    assert found and loc["method"] == "exact"
    s, e = loc["char"]
    assert DOC[s:e].startswith("virtual channel unit configured to time-share")


def test_snap_fuzzy_small_edit():
    q = "the switching unit stores that data in the second receive buffer when the link partner lacks VC2 support"
    found, loc = snap_quote(q, DOC)
    assert found and loc["method"] in ("exact", "fuzzy")
    s, e = loc["char"]
    assert "switching unit" in DOC[s:e]


def test_snap_dual_local_paraphrase():
    q = "the switching unit stores the data in the second receive buffer if the link partner lacks support"
    found, loc = snap_quote(q, DOC)
    assert found and loc["method"] == "dual"
    s, e = loc["char"]
    assert "receive buffer" in DOC[s:e]


def test_snap_fabricated_fails():
    found, loc = snap_quote("a convolutional encoder compresses the video stream before transmission", DOC)
    assert not found and loc is None


def test_snap_too_short_fails():
    assert snap_quote("bus", DOC) == (False, None)
