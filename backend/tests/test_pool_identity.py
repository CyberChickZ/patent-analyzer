import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.recall.pool import Candidate


def test_same_title_different_patents_are_not_merged():
    from patent_analyzer.recall.pool import pool_and_dedupe
    t = "Multiple sensor fish surrogate for acoustic and hydraulic data collection"
    a = Candidate(title=t, pub_num="US5675555A", match_type="Patent", raw={})
    b = Candidate(title=t, pub_num="US5517465A", match_type="Patent", raw={})
    c = Candidate(title=t, pub_num="", match_type="Paper", raw={})
    out = pool_and_dedupe({"x": [a, b, c]})
    assert sorted(x.pub_num for x in out) == ["", "US5517465A", "US5675555A"]
