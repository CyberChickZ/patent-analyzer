import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm import _align_checklist_keys

CL = [{"criterion": "A data transfer apparatus comprising:"},
      {"criterion": "a virtual channel unit configured to time share a serial bus"},
      {"criterion": "a switching unit configured to control storing the data"}]


def test_numeric_labels_map_to_criteria():
    r = {"checklist_results": {"1": {"score": 2}, "2.": {"score": 1}, "3": {"score": 0}}}
    assert _align_checklist_keys(r, CL) == 0
    assert r["checklist_results"][CL[0]["criterion"]]["score"] == 2
    assert r["checklist_results"][CL[1]["criterion"]]["score"] == 1


def test_exact_keys_pass_through():
    r = {"checklist_results": {CL[2]["criterion"]: {"score": 2}}}
    assert _align_checklist_keys(r, CL) == 0
    assert list(r["checklist_results"]) == [CL[2]["criterion"]]


def test_paraphrased_key_fuzzy_aligns():
    r = {"checklist_results": {"A virtual channel unit configured to time-share a serial bus.": {"score": 2}}}
    assert _align_checklist_keys(r, CL) == 0
    assert CL[1]["criterion"] in r["checklist_results"]


def test_unrelated_key_counted_unmatched():
    r = {"checklist_results": {"quantum encryption layer": {"score": 2}}}
    assert _align_checklist_keys(r, CL) == 1
    assert "quantum encryption layer" in r["checklist_results"]


def test_out_of_range_number_is_unmatched():
    r = {"checklist_results": {"9": {"score": 2}}}
    assert _align_checklist_keys(r, CL) == 1
