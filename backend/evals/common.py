"""Shared helpers for the stage-level evals (FiNE-Patents adapters)."""

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "eval_data" / "fine-patents" / "data" / "packaged"
FIXTURE_DIR = Path(__file__).parent / "fixtures"


def render_doc(patent: dict, with_claims: bool) -> str:
    """Render a FiNE patent json as the plain-text document a user would upload.

    First line is `Title: ...` so nodes/idca.py picks up source_title. Description
    paragraphs keep their [000n] markers. with_claims=False simulates a paper /
    technical report (the real product scenario); True is the patent-draft
    upper bound.
    """
    title = (patent.get("title") or "").strip()
    abstract = (patent.get("abstract") or "").strip()
    paragraphs = [p for p in (patent.get("description") or []) if p]
    parts = [f"Title: {title}", ""]
    if abstract:
        parts += ["Abstract", abstract, ""]
    parts += ["Description"] + paragraphs
    if with_claims:
        claims = [c for c in (patent.get("claims") or []) if c]
        parts += ["", "Claims:"] + claims
    return "\n".join(parts).strip() + "\n"
