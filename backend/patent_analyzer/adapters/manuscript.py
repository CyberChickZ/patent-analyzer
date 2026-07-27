"""Manuscript adapter: a submission draft is a paper whose Related Work /
Background must not be mistaken for the authors' own invention. Those
sections are dropped and the abstract is cleared so A1/A2 only see
Method / Results / Discussion."""

import copy
import re

RELATED_WORK = re.compile(
    r"^\s*(?:[\dIVX]+(?:\.\d+)*\.?\s+)?"
    r"(?:related\s+works?|background(?:\s+and\s+related\s+work)?|prior\s+art|"
    r"literature\s+(?:review|survey)|previous\s+works?|state\s+of\s+the\s+art)\b",
    re.IGNORECASE)


def is_related_work_title(title: str) -> bool:
    return bool(RELATED_WORK.match(title or ""))


def strip_related_work(doc: dict) -> dict:
    """Return a copy of `doc` (kind='manuscript') without related-work sections
    at any depth and with an empty abstract. `dropped_sections` lists the titles removed."""
    out = copy.deepcopy(doc)
    dropped: list[str] = []

    def prune(secs):
        kept = []
        for s in secs or []:
            if is_related_work_title(s.get("title", "")):
                dropped.append(s.get("title", ""))
                continue
            s["subsections"] = prune(s.get("subsections"))
            kept.append(s)
        return kept

    out["sections"] = prune(out.get("sections"))
    out["abstract"] = ""
    out["kind"] = "manuscript"
    out["dropped_sections"] = dropped
    return out
