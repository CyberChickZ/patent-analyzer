import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import fulltext_gap as gap  # noqa: E402


def _results(rows, manifest=None, delivered=None):
    return {
        "job_id": "jg",
        "phase1": {"summary": "S"},
        "phase2": {"checklist": [{"id": "c1", "criterion": "A"}]},
        "search": {"summary": {"fulltext_oa": {"manifest": manifest or []}}},
        "eval_stats": {"read_gap": {"delivered": delivered or len(rows)}},
        "adjudication": {"label": "allow", "label_text": "No blocking reference"},
        "evaluation": {"scoring_report": rows},
    }


# ── which references are listed ──────────────────────────────────────────────

def test_a_reference_read_from_its_pdf_is_not_on_the_list():
    rows = gap.gap_rows(_results([{"pub_num": "10.1/a", "title": "A", "source": "pdf"}]))
    assert rows == []


def test_claims_from_bigquery_also_count_as_read():
    rows = gap.gap_rows(_results([{"pub_num": "US-1-A", "title": "P", "source": "full_text"}]))
    assert rows == []


def test_abstract_only_and_no_content_are_both_listed_and_distinguished():
    rows = gap.gap_rows(_results([
        {"pub_num": "10.1/a", "title": "A", "source": "abstract", "text_chars": 900, "similarity_score": 0.4},
        {"pub_num": "10.1/b", "title": "B", "source": "no_content",
         "no_content_reason": "no PDF and no abstract", "similarity_score": 0.1},
    ]))
    assert [r["read_state"] for r in rows] == ["abstract_only", "nothing"]
    assert rows[1]["read_reason"] == "no PDF and no abstract"


def test_the_source_duplicate_is_not_a_missing_reference():
    rows = gap.gap_rows(_results([{"pub_num": "10.1/a", "title": "A", "source": "no_content",
                                   "is_source_duplicate": True}]))
    assert rows == []


def test_highest_scoring_first_because_that_one_can_still_change_the_verdict():
    rows = gap.gap_rows(_results([
        {"pub_num": "10.1/low", "title": "low", "source": "abstract", "similarity_score": 0.1},
        {"pub_num": "10.1/high", "title": "high", "source": "abstract", "similarity_score": 0.8},
    ]))
    assert [r["pub_num"] for r in rows] == ["10.1/high", "10.1/low"]


# ── the tier trail ───────────────────────────────────────────────────────────

def test_the_trail_names_every_tier_and_why_it_gave_up():
    rows = gap.gap_rows(_results([{
        "pub_num": "10.1/a", "title": "A", "source": "abstract",
        "fulltext_tier": "abstract_only", "fulltext_detail": "Unpaywall: not open access",
    }]))
    trail = {a["tier"]: a for a in rows[0]["attempts"]}
    assert trail["arxiv"]["outcome"] == "missed" and "no arXiv id" in trail["arxiv"]["detail"]
    assert trail["oa"]["outcome"] == "failed" and trail["oa"]["detail"] == "Unpaywall: not open access"
    assert trail["pdf_download"]["outcome"] == "skipped"


def test_a_resolved_url_that_produced_nothing_is_a_failed_download_not_a_missing_one():
    rows = gap.gap_rows(_results([{
        "pub_num": "10.1/a", "title": "A", "source": "abstract",
        "fulltext_tier": "abstract_only", "fulltext_url": "https://example.org/x.pdf",
        "fulltext_detail": "Unpaywall (repository) — resolved, but the URL returned no PDF",
    }]))
    dl = next(a for a in rows[0]["attempts"] if a["tier"] == "pdf_download")
    assert dl["outcome"] == "failed"


def test_the_explicit_download_stamp_wins_over_the_inference():
    rows = gap.gap_rows(_results([{
        "pub_num": "10.1/a", "title": "A", "source": "abstract", "fulltext_tier": "oa",
        "fulltext_url": "https://example.org/x.pdf", "fulltext_download": "skipped_budget",
    }]))
    dl = next(a for a in rows[0]["attempts"] if a["tier"] == "pdf_download")
    assert dl["outcome"] == "skipped" and "budget" in dl["detail"]


def test_a_job_that_predates_the_tiers_says_unknown_rather_than_inventing_a_failure():
    rows = gap.gap_rows(_results([{"pub_num": "10.1/a", "title": "A", "source": "abstract"}]))
    outcomes = {a["tier"]: a["outcome"] for a in rows[0]["attempts"]}
    assert outcomes["arxiv"] == "unknown" and outcomes["oa"] == "unknown"


def test_a_patent_trail_starts_at_our_own_claims_table():
    rows = gap.gap_rows(_results([{"pub_num": "US-9999999-B2", "title": "P", "source": "no_content",
                                   "had_bq_claims": False}]))
    first = rows[0]["attempts"][0]
    assert first["tier"] == "bigquery_claims" and first["outcome"] == "missed"


# ── identifiers ──────────────────────────────────────────────────────────────

def test_the_doi_is_recovered_from_pub_num_when_the_row_has_no_doi_field():
    rows = gap.gap_rows(_results([{"pub_num": "10.48550/arXiv.2607.28443", "title": "A", "source": "abstract"}]))
    assert rows[0]["doi"] == "10.48550/arxiv.2607.28443"
    assert rows[0]["landing_page"] == "https://doi.org/10.48550/arxiv.2607.28443"


def test_the_manual_manifest_fills_in_a_row_the_eval_never_stamped():
    rows = gap.gap_rows(_results(
        [{"pub_num": "", "title": "Some paper", "source": "abstract"}],
        manifest=[{"doi": "10.1/m", "title": "Some paper", "landing_page": "https://p/x",
                   "reason": "Unpaywall: DOI unknown"}]))
    assert rows[0]["doi"] == "10.1/m" and rows[0]["landing_page"] == "https://p/x"
    assert next(a for a in rows[0]["attempts"] if a["tier"] == "oa")["detail"] == "Unpaywall: DOI unknown"


def test_ref_id_is_stable_and_url_safe():
    assert gap.ref_id({"pub_num": "10.1109/TNNLS.2025.3644299"}) == "10.1109-tnnls.2025.3644299"
    assert gap.ref_id({"title": "A Paper: Part 2"}) == "a-paper-part-2"


def test_the_summary_counts_add_up_to_what_was_evaluated():
    res = _results([{"pub_num": "a", "title": "a", "source": "pdf"},
                    {"pub_num": "b", "title": "b", "source": "abstract"},
                    {"pub_num": "c", "title": "c", "source": "no_content"}])
    s = gap.gap_summary(res, gap.gap_rows(res))
    assert s == {"evaluated": 3, "missing": 2, "abstract_only": 1, "nothing": 1,
                 "uploaded": 0, "pending_reread": 0, "full_text": 1}
