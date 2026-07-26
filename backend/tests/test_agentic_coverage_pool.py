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
