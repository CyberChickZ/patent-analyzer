"""No counter may live only in this container.

Harry, 2026-09-20: "不能有 local counter，必须云端". The rule, stated so a
grep can check it:

  State that outlives a request goes to GCS (patent_analyzer.cloud_state).
  A container-local store may hold a CACHE — something whose loss costs time
  and nothing else.

The guard is deliberately crude. It cannot tell a counter from a cache by
reading it, so it checks the two things that are checkable: that the modules
which hold cross-request state do not reach for the container-local KV, and
that nothing user-facing calls a number a "local counter" again.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PKG = Path(__file__).parent.parent / "patent_analyzer"

#: Modules that hold state which must survive a request and be shared.
STATEFUL = ["runtime_state.py", "spend.py"]

#: Modules whose kv() use is a cache: losing it costs a repeated call, never
#: correctness. Each one is listed on purpose — a new name has to be argued
#: for here rather than added quietly.
CACHES = {
    "fulltext.py": "Unpaywall / Europe PMC responses",
    "fulltext_sources.py": "Crossref, PMCID and DOI lookups",
    "recall/lens.py": "Lens search results",
    "recall/google_patents.py": "Google Patents search results",
    "recall/openalex.py": "title to DOI",
    "recall/serpapi.py": "search results, and the /account sync timestamp",
    "recall/uspto_odp.py": "ODP responses",
    "quota.py": "the 60s sync timestamp and the BigQuery month cache",
    "metering.py": "the month-to-date BigQuery tally, mirrored from INFORMATION_SCHEMA",
}


def test_state_modules_do_not_use_the_container_local_kv():
    bad = []
    for name in STATEFUL:
        src = (PKG / name).read_text()
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        # the docstring explains why it does not; the code must not call it
        if re.search(r"\bkv\(\)\s*\.", code):
            bad.append(name)
    assert not bad, f"these hold cross-request state and must use cloud_state, not kv(): {bad}"


def test_every_kv_user_is_a_declared_cache():
    """A module that starts using kv() has to say which kind it is."""
    users = set()
    for p in PKG.rglob("*.py"):
        if p.name in ("cache.py", "cloud_state.py"):
            continue
        if re.search(r"\bkv\(\)\s*\.", p.read_text()):
            users.add(str(p.relative_to(PKG)))
    undeclared = sorted(users - set(CACHES) - set(STATEFUL))
    assert not undeclared, (
        "new kv() users must be declared in CACHES (loss costs time only) or moved to "
        f"cloud_state (loss costs correctness): {undeclared}")


def test_nothing_reports_a_number_as_a_local_counter():
    """The panel used to label our own tallies "local counter", which was both
    true and the problem. They are in GCS now and say so."""
    from patent_analyzer import quota
    assert quota.BASIS_OURS == "our count, stored in cloud"
    written = [v for k, v in vars(quota).items()
               if k.startswith("BASIS_") and k != "BASIS_LOCAL"]
    assert "local counter" not in written


def test_the_quota_panel_never_emits_the_retired_label():
    import asyncio

    from patent_analyzer import quota
    snap = asyncio.run(quota.snapshot())
    bad = [r["name"] for r in snap["sources"] if r["basis"] == quota.BASIS_LOCAL]
    assert not bad, f"still reporting a local counter: {bad}"
