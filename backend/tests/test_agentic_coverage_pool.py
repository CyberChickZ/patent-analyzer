import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.coverage import tag_coverage
from patent_analyzer.recall.pool import Candidate, pool_and_dedupe

ELS = [{"id": "e1", "text": "gaze estimation for video conferencing", "facets": {"thing": ["gaze"], "place": ["conferenc"]}},
       {"id": "e2", "text": "quantum tunneling widget", "facets": {"thing": ["quantum"], "place": []}}]
DOCS = [{"pub_num": "US1", "title": "Horizontal gaze estimation for video conferencing", "abstract": ""},
        {"pub_num": "US2", "title": "Eye tracker", "abstract": "gaze is tracked"},
        {"pub_num": "US3", "title": "Unrelated", "abstract": "nothing here"}]


def test_lexical_coverage_requires_thing_and_place():
    cov = tag_coverage(ELS, DOCS)
    assert cov["e1"] == ["US1"]      # US2 has 'gaze' but no 'conferenc'
    assert cov["e2"] == []


def test_semantic_coverage_adds_hits():
    def embed(texts):
        return np.array([[1.0, 0.0] if "quantum" in t else [0.0, 1.0] for t in texts])
    docs = DOCS + [{"pub_num": "US4", "title": "quantum device", "abstract": ""}]
    cov = tag_coverage(ELS, docs, embed=embed, tau=0.9)
    assert "US4" in cov["e2"]


def test_pool_dedupes_by_family_before_pub():
    a = Candidate(title="Patent A long enough title here", pub_num="US1B2", match_type="Patent",
                  raw={"bigquery": {"family_id": "F1"}})
    b = Candidate(title="Patent A translated title différent", pub_num="EP1A1", match_type="Patent",
                  raw={"bigquery": {"family_id": "F1"}})
    c = Candidate(title="Other patent unrelated title text", pub_num="US9B2", match_type="Patent",
                  raw={"bigquery": {"family_id": "F2"}})
    pooled = pool_and_dedupe({"gp": [a], "cit": [b, c]})
    assert len(pooled) == 2
    fam1 = next(p for p in pooled if p.pub_num == "US1B2")
    assert sorted(fam1.sources) == ["cit", "gp"]


def test_merge_facets_unions_two_samples_in_order():
    from patent_analyzer.agentic.elements import merge_facets
    m = merge_facets({"thing": ["gaze estimation", "eye tracking"], "place": ["telepresence"]},
                     {"thing": ["Eye Tracking", "gaze direction", "line of sight"], "place": [], "apparatus": ["camera"]})
    assert m == {"named": [], "thing": ["gaze estimation", "eye tracking", "gaze direction", "line of sight"],
                 "place": ["telepresence"], "apparatus": ["camera"]}
    assert len(merge_facets({"thing": [str(i) for i in range(20)]}, {})["thing"]) == 10


def test_attach_facets_widens_every_element(monkeypatch):
    import asyncio
    from patent_analyzer.agentic.elements import attach_facets
    asked = []

    async def fake(els, summary):
        asked.extend(e["id"] for e in els)
        return {"e1": {"thing": ["eye tracking", "gaze direction"], "place": ["video call"], "apparatus": []},
                "e2": {"thing": [], "place": [], "apparatus": []}}
    monkeypatch.setattr("app.llm.facet_elements", fake)
    els = [{"id": "e1", "text": "gaze estimation for telepresence", "facets": {"thing": ["gaze estimation"], "place": ["telepresence"]}},
           {"id": "e2", "text": "a motorized swivelling turntable", "facets": {}}]
    out = asyncio.run(attach_facets(els, "s"))
    assert asked == ["e1", "e2"]
    assert out[0]["facets"]["thing"] == ["gaze estimation", "eye tracking", "gaze direction"]
    assert out[0]["facets"]["place"] == ["telepresence", "video call"]
    assert out[1]["facets"]["thing"]  # fallback kept the loop alive



def test_valid_name_requires_source_presence_and_distinctiveness():
    from patent_analyzer.agentic.elements import merge_facets, valid_name
    src = "We inject indocyanine green (ICG) intravenously. A tablet PC shows the feed. The PSS uses two CCD cameras."
    assert valid_name("indocyanine green", src) and valid_name("icg", src)
    assert not valid_name("tablet pc", src)   # generic product category
    assert valid_name("ccd", src)             # capitalised in the source: kept (generic list has no 'ccd')
    assert valid_name("pss", src)            # capitalised acronym present in the source
    assert not valid_name("teleconferencing robot", src)   # not in the source
    m = merge_facets({"named": ["tablet pc", "icg"], "thing": ["perfusion map"]},
                     {"patent": ["tissue perfusion imaging"], "named": ["indocyanine green", "made up name"], "thing": ["blood flow map"]},
                     source=src)
    assert m["named"] == ["icg", "indocyanine green"]
    assert m["thing"] == ["tissue perfusion imaging", "perfusion map", "blood flow map"]
