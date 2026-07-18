"""Offline tests for the eval adapters (evals/common.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from common import norm_pub, render_doc


def test_norm_pub_fine_spelling():
    assert norm_pub("US 2008/025717 A1") == "US20080025717A1"


def test_norm_pub_bigquery_spelling():
    assert norm_pub("US-2008025717-A1") == "US20080025717A1"


def test_norm_pub_already_normalized():
    assert norm_pub("US20080025717A1") == "US20080025717A1"


def test_norm_pub_granted_and_ep():
    assert norm_pub("US 7,123,456 B2") == "US7123456B2"
    assert norm_pub("EP 1 234 567 A1") == "EP1234567A1"
    assert norm_pub("wo2019/123456a1") == "WO2019123456A1"


def test_norm_pub_empty():
    assert norm_pub("") == ""


def test_render_doc_desc_only_has_no_claims():
    doc = {"title": "T", "abstract": "Abs", "description": ["[0001] one", None, "[0002] two"],
           "claims": ["1. A thing."]}
    txt = render_doc(doc, with_claims=False)
    assert txt.startswith("Title: T\n")
    assert "[0002] two" in txt
    assert "Claims:" not in txt and "A thing" not in txt


def test_render_doc_with_claims_appends_claims():
    doc = {"title": "T", "abstract": "", "description": ["[0001] one"], "claims": ["1. A thing."]}
    txt = render_doc(doc, with_claims=True)
    assert "Abstract" not in txt
    assert txt.rstrip().endswith("Claims:\n1. A thing.")
