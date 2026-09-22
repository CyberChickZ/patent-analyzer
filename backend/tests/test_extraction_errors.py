import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from extraction_errors import classify_errors, error_rates, snap_quote

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


def _fake_embed(vocab):
    def embed(texts):
        out = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, w in enumerate(vocab):
                if w in t.lower():
                    out[i, j] = 1.0
            n = np.linalg.norm(out[i]) or 1.0
            out[i] /= n
        return out
    return embed


def test_classify_errors_three_buckets():
    embed = _fake_embed(["virtual channel", "time-share", "buffering", "receive buffer",
                         "hexagonal", "coils", "encoder", "video"])
    gold = [
        {"text": "a virtual channel unit that time-shares the bus", "kind": "structure"},
        {"text": "a buffering control unit", "kind": "structure"},
        {"text": "coils arranged in a hexagonal lattice", "kind": "structure"},
    ]
    pred = [
        {"id": "e1", "text": "virtual channel unit configured to time-share the bus", "kind": "step",
         "evidence_quote": "a virtual channel unit configured to time-share a serial bus"},
        {"id": "e2", "text": "a buffering control unit receiving data", "kind": "structure",
         "evidence_quote": "A buffering control unit receives data via the first virtual channel"},
        {"id": "e3", "text": "a video encoder compressing the stream", "kind": "structure",
         "evidence_quote": "a convolutional encoder compresses the video stream"},
    ]
    err = classify_errors(pred, gold, DOC, embed_fn=embed)
    assert err["omission"] == [2]
    assert err["fabrication"] == ["e3"]
    assert [m["pred"] for m in err["misclassification"]] == ["e1"]
    assert err["misclassification"][0]["attr"] == "kind"
    assert {(g, p) for g, p, _ in err["matched"]} == {(0, 0), (1, 1)}
    assert err["unsupported"] == ["e3"]
    assert abs(err["quote_survival"] - 2 / 3) < 1e-6
    rates = error_rates(err)
    assert abs(rates["omission"] - 1 / 3) < 1e-6
    assert abs(rates["fabrication"] - 1 / 3) < 1e-6
    assert abs(rates["misclassification"] - 1 / 2) < 1e-6


def test_classify_errors_orphan_with_real_quote_is_not_fabrication_when_on_topic():
    embed = _fake_embed(["virtual channel", "hexagonal", "coils", "lattice"])
    gold = [{"text": "a virtual channel unit"}]
    pred = [
        {"id": "e1", "text": "a virtual channel unit", "evidence_quote": "a virtual channel unit configured to time-share"},
        {"id": "e2", "text": "coils in a hexagonal lattice",
         "evidence_quote": "Wireless power transfer coils are arranged in a hexagonal lattice"},
    ]
    err = classify_errors(pred, gold, DOC, embed_fn=embed)
    assert err["fabrication"] == []
    assert err["omission"] == []


def test_classify_errors_level_mismatch_and_no_quote_mode():
    embed = _fake_embed(["virtual channel", "coils"])
    gold = [{"text": "a virtual channel unit", "level": "core"}]
    pred = [{"id": "e1", "text": "a virtual channel unit", "level": "component"}]
    err = classify_errors(pred, gold, DOC, embed_fn=embed, require_quote=False)
    assert err["fabrication"] == []
    assert err["misclassification"][0]["attr"] == "level"
    assert err["quote_survival"] is None


def test_classify_errors_empty_inputs():
    err = classify_errors([], ["a gold feature"], DOC, embed_fn=_fake_embed(["x"]))
    assert err["omission"] == [0] and err["fabrication"] == [] and err["matched"] == []
