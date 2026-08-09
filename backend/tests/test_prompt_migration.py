"""The registry templates must render exactly what the f-strings produced
before the migration (commit 0eb85c3 is the last f-string version)."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm, prompts

OLD_REV = "0eb85c3"


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


def test_search_facets_template_matches_old_fstring():
    fs = _old_fstring('    prompt = f"""For EACH element below, give four facets of search terms:')
    summary, listing = "S" * 5000, "e1: a\ne2: b"
    expected = eval(fs, {"summary": summary, "listing": listing})
    assert prompts.render("search.facets", summary=summary[:4000], listing=listing) == expected
