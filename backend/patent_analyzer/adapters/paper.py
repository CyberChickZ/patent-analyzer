"""Paper adapter: plain text -> Doc, and Doc -> text with [S1.P3] paragraph markers.

Doc = {title, abstract, sections: [{title, paragraphs: [str], subsections: [...]}], kind}
(Pap2Pat's paper.json shape). Markers let a char offset in the rendered text
be mapped back to (section path, paragraph index) for evidence_loc.
"""

import re

_MD_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUM_HEAD = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.]{1,90})$")
_ROMAN_HEAD = re.compile(r"^(?:[IVX]+\.)\s+([A-Z][^.]{1,90})$")
_PARA_MARK = re.compile(r"^\[\d{3,5}\]")
_MARKER = re.compile(r"\[(S[\d.]+)\.P(\d+)\]")
_ABSTRACT = re.compile(r"^abstract(?:\s*[:—–-]\s*(.*))?$", re.IGNORECASE)
_WORD_HEAD = re.compile(r"^[A-Z][a-z]{2,29}$")


def _caps_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return (3 <= len(letters) and len(line) <= 80 and not line.endswith(".")
            and sum(c.isupper() for c in letters) / len(letters) >= 0.9)


def _heading(line: str) -> tuple[int, str] | None:
    m = _MD_HEAD.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _NUM_HEAD.match(line)
    if m and len(line) <= 100:
        return m.group(1).count(".") + 1, line.strip()
    m = _ROMAN_HEAD.match(line)
    if m:
        return 1, line.strip()
    if _caps_heading(line) or _WORD_HEAD.match(line):
        return 1, line.strip()
    return None


def _split_paragraphs(lines: list[str]) -> list[str]:
    """Blank lines separate paragraphs; a [0001]-style marker always starts one.
    Text with almost no blank lines (one paragraph per line) is split per line."""
    n_blank = sum(1 for l in lines if not l.strip())
    n_text = len(lines) - n_blank
    per_line = n_text > 8 and n_blank < n_text / 8
    out, buf = [], []

    def flush():
        if buf:
            out.append(" ".join(buf).strip())
            buf.clear()

    for l in lines:
        s = l.strip()
        if not s or per_line or _PARA_MARK.match(s):
            flush()
        if s:
            buf.append(s)
    flush()
    return [p for p in out if p]


def _new_section(title: str) -> dict:
    return {"title": title, "paragraphs": [], "subsections": []}


def doc_from_text(text: str, kind: str = "paper") -> dict:
    """Cut plain text into a Doc. Headings: markdown '#', numbered ('3.2 Method'),
    roman ('II. Related Work') and short ALL-CAPS lines. 'Title:' first line and
    an 'Abstract' heading are lifted into their own fields."""
    doc = {"title": "", "abstract": "", "sections": [], "kind": kind}
    lines = (text or "").replace("\r\n", "\n").split("\n")
    body_start = 0
    for i, l in enumerate(lines[:3]):
        if l.strip().lower().startswith("title:"):
            doc["title"] = l.strip()[6:].strip()
            body_start = i + 1
            break

    stack: list[tuple[int, dict]] = []
    preface: dict = _new_section("")
    current_lines: list[str] = []
    in_abstract = False

    def close_block():
        nonlocal in_abstract
        paras = _split_paragraphs(current_lines)
        current_lines.clear()
        if in_abstract:
            doc["abstract"] = " ".join(paras).strip()
            in_abstract = False
            return
        target = stack[-1][1] if stack else preface
        target["paragraphs"].extend(paras)

    for l in lines[body_start:]:
        s = l.strip()
        am = _ABSTRACT.match(s)
        if am and not stack and not doc["abstract"]:
            close_block()
            in_abstract = True
            if am.group(1) and am.group(1).strip():
                current_lines.append(am.group(1).strip())
            continue
        h = _heading(s)
        if h:
            close_block()
            depth, title = h
            sec = _new_section(title)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            (stack[-1][1]["subsections"] if stack else doc["sections"]).append(sec)
            stack.append((depth, sec))
            continue
        current_lines.append(l)
    close_block()
    if preface["paragraphs"]:
        doc["sections"].insert(0, preface)
    return doc


def doc_from_sections(title: str, abstract: str, sections: list[dict], kind: str = "paper") -> dict:
    """Wrap an already-structured paper (Pap2Pat paper.json) as a Doc."""
    def clean(secs):
        return [{"title": s.get("title") or "", "paragraphs": [p for p in (s.get("paragraphs") or []) if p],
                 "subsections": clean(s.get("subsections") or [])} for s in secs or []]
    return {"title": title or "", "abstract": abstract or "", "sections": clean(sections), "kind": kind}


def render_doc(doc: dict) -> str:
    """Doc -> text. Each paragraph is prefixed by [S<path>.P<n>]; the abstract is S0."""
    parts = []
    if doc.get("title"):
        parts += [f"Title: {doc['title']}", ""]
    if doc.get("abstract"):
        parts += ["Abstract", f"[S0.P1] {doc['abstract']}", ""]

    def walk(secs, path, depth):
        for i, s in enumerate(secs or [], 1):
            sp = f"{path}.{i}" if path else f"S{i}"
            if s.get("title"):
                parts.append("#" * depth + " " + s["title"])
            for j, p in enumerate(s.get("paragraphs") or [], 1):
                parts.append(f"[{sp}.P{j}] {p}")
            parts.append("")
            walk(s.get("subsections"), sp, depth + 1)

    walk(doc.get("sections"), "", 1)
    return "\n".join(parts).strip() + "\n"


def locate_marker(rendered: str, char_pos: int) -> dict:
    """(section, para) of the last [S..P..] marker before char_pos; None if none."""
    last = None
    for m in _MARKER.finditer(rendered):
        if m.start() > char_pos:
            break
        last = m
    if not last:
        return {"section": None, "para": None}
    return {"section": last.group(1), "para": int(last.group(2))}


def iter_paragraphs(doc: dict):
    """Yield (section_path, para_idx, text) in render order (abstract = S0)."""
    if doc.get("abstract"):
        yield "S0", 1, doc["abstract"]

    def walk(secs, path):
        for i, s in enumerate(secs or [], 1):
            sp = f"{path}.{i}" if path else f"S{i}"
            for j, p in enumerate(s.get("paragraphs") or [], 1):
                yield sp, j, p
            yield from walk(s.get("subsections"), sp)

    yield from walk(doc.get("sections"), "")
