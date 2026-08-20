"""The registry templates must render exactly what the f-strings produced
before the migration (commit 0eb85c3 is the last f-string version)."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm, prompts

OLD_REV = "0eb85c3"          # last f-string version of extract.* / search.facets
EVAL_REV = "584c5fc"         # last f-string version of evaluate.*


def _old_fstring(marker: str) -> str:
    src = subprocess.run(["git", "show", f"{OLD_REV}:backend/app/llm.py"], capture_output=True, text=True,
                         cwd=Path(__file__).parent.parent).stdout
    if not src:
        pytest.skip("old revision not available")
    i = src.index(marker)
    j = src.index('"""', i + len(marker) + 5) + 3
    return src[i:j][len("    prompt = "):]


def test_extract_elements_template_matches_old_fstring():
    fs = _old_fstring('    prompt = f"""════ TASK ════\nFor EACH candidate invention below')
    cand_lines, prefill_block, feedback, doc_text = "- inv1 [core]: thing", "\n\n════ PREFILLED ELEMENTS (FIXED) ════\nx\n", {"note": "n"}, "DOC " * 10
    expected = eval(fs, {"_feedback_block": llm._feedback_block, "_EXTRACTION_DOC_CAP": llm._EXTRACTION_DOC_CAP,
                         "cand_lines": cand_lines, "prefill_block": prefill_block, "feedback": feedback, "doc_text": doc_text})
    got = prompts.render("extract.elements", cand_lines=cand_lines, prefill_block=prefill_block,
                         feedback_block=llm._feedback_block(feedback), document=doc_text[:llm._EXTRACTION_DOC_CAP])
    assert got == expected


def test_search_facets_template_matches_old_fstring_except_the_patent_facet():
    # H7 added the `patent` facet on purpose; the rest of the template is unchanged
    fs = _old_fstring('    prompt = f"""For EACH element below, give four facets of search terms:')
    summary, listing = "S" * 5000, "e1: a\ne2: b"
    expected = eval(fs, {"summary": summary, "listing": listing})
    got = prompts.render("search.facets", summary=summary[:4000], listing=listing)
    assert got.split("  named     —", 1)[1].replace('"patent": [...], ', "") == expected.split("  named     —", 1)[1]
    assert got.startswith("For EACH element below, give five facets") and "patent    —" in got


def test_extract_candidates_template_matches_old_fstring():
    src = subprocess.run(["git", "show", f"{OLD_REV}:backend/app/llm.py"], capture_output=True, text=True, cwd=Path(__file__).parent.parent).stdout
    i = src.index("async def extract_candidates")
    i = src.index('    prompt = f"""', i)
    j = src.index('"""', i + 20) + 3
    fs = src[i:j][len("    prompt = "):]
    doc_kind, guidance, summary, doc_text = "paper", llm._DOC_KIND_GUIDANCE["paper"], "S" * 5000, "D" * 20
    expected = eval(fs, {"_EXTRACTION_DOC_CAP": llm._EXTRACTION_DOC_CAP, "doc_kind": doc_kind, "guidance": guidance, "summary": summary, "doc_text": doc_text})
    assert prompts.render("extract.candidates", doc_kind=doc_kind, guidance=guidance, summary=summary[:4000], document=doc_text) == expected


def test_idca_summarize_registered_verbatim():
    src = subprocess.run(["git", "show", f"{OLD_REV}:backend/app/llm.py"], capture_output=True, text=True, cwd=Path(__file__).parent.parent).stdout
    i = src.index('    task_prompt = """════ TASK ════\nRead the ENTIRE')
    j = src.index('"""', i + 30) + 3
    assert prompts.get("idca.summarize")[0] == src[i:j][len('    task_prompt = """'):-3]


def _fstring_in(fn_name: str, start_marker: str, rev: str = EVAL_REV) -> str:
    src = subprocess.run(["git", "show", f"{rev}:backend/app/llm.py"], capture_output=True, text=True, cwd=Path(__file__).parent.parent).stdout
    L = src.index(fn_name)
    i = src.index(start_marker, L)
    j = src.index('"""', i + 30) + 3
    return src[i:j][len("    prompt = "):]


def test_evaluate_templates_match_old_fstrings():
    fs = _fstring_in("async def evaluate_single_document_text", '    prompt = f"""INVENTION: {invention_summary}')
    kw = dict(invention_summary="INV", checklist=[1, 2, 3], cl_text="CL", prior_art_title="T", prior_art_type="Patent",
              doc_label="full text", prior_art_text="DOC", scoring_instruction="SI", output_schema='"x": 1')
    assert prompts.render("evaluate.document_text", n_items=3, **{k: v for k, v in kw.items() if k != "checklist"}) == eval(fs, dict(kw))
    fs = _fstring_in("async def evaluate_single_document(", '    prompt = f"""{source_label}{pa_ordinal} attached PDF')
    kw = dict(source_label="First ", pa_ordinal="second", fig_note="FN", invention_summary="INV", checklist=[1, 2], cl_text="CL",
              prior_art_title="T", prior_art_type="Patent", scoring_instruction="SI", output_schema='"x": 1')
    assert prompts.render("evaluate.document_pdf", n_items=2, **{k: v for k, v in kw.items() if k != "checklist"}) == eval(fs, dict(kw))
