import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from quote_dual import DocIndex, locate_dual, span_ratio, verify_quote_dual

DOC = ("[0001] The apparatus comprises a virtual channel unit configured to time-share "
       "a serial bus between a first virtual channel and a second virtual channel.\n"
       "[0002] A buffering control unit receives data via the first virtual channel; "
       "the switching unit stores that data in the second receive buffer when the "
       "link partner lacks VC1 support.")
REFS = [("paragraph", 1, DOC.split("\n")[0]), ("paragraph", 2, DOC.split("\n")[1]),
        ("abstract", None, "A bus arbitration scheme for wireless power transfer coils.")]


def test_exact_quote_passes_both():
    ok, sr, br = verify_quote_dual("a buffering control unit receives data via the first virtual channel", DOC)
    assert ok and sr == 1.0 and br == 1.0


def test_local_paraphrase_passes_span_and_enough_bigrams():
    # words reordered / one substituted inside one sentence: fuzzy difflib would fail
    ok, sr, br = verify_quote_dual(
        "the switching unit stores the data in the second receive buffer if the link partner lacks support", DOC)
    assert ok and sr >= 0.7 and br >= 0.3


def test_bag_of_words_from_far_apart_spans_rejected():
    # on-topic words scattered across the document, never adjacent in this order
    ok, sr, br = verify_quote_dual("buffer serial partner link apparatus unit channel stores time", DOC)
    assert not ok and br < 0.3


def test_fabricated_quote_rejected():
    ok, sr, _ = verify_quote_dual("the apparatus uses quantum tunneling to encrypt every packet", DOC)
    assert not ok and sr < 0.7


def test_too_few_content_words_rejected():
    assert verify_quote_dual("the serial bus", DOC) == (False, 0.0, 0.0)


def test_span_ratio_sliding_window():
    doc = "x y z a b c x y z x y z d e f".split()
    assert span_ratio(["a", "b", "c"], doc, window=3) == 1.0
    assert span_ratio(["a", "d"], doc, window=3) == 0.5


def test_docindex_reuse_matches_string_path():
    idx = DocIndex(DOC)
    q = "virtual channel unit configured to time-share a serial bus"
    assert verify_quote_dual(q, idx) == verify_quote_dual(q, DOC)


def test_locate_dual_picks_passage():
    assert locate_dual("switching unit stores that data in the second receive buffer", REFS) == ("paragraph", 2)
    assert locate_dual("virtual channel unit configured to time-share a serial bus", REFS) == ("paragraph", 1)
    assert locate_dual("quantum tunneling encrypts every packet on the bus", REFS) is None
