"""112(b) self-check by rule (leader_draft §2.4). Four PEDANTIC categories the
office's reasons fall into ~57% of the time are checkable without an LLM:

  antecedent_basis     MPEP 2173.05(e): "the/said X" with no earlier "a X"
                       (or an earlier "a X" only, when "the <adj> X" is used),
                       or two earlier introductions of the same X
  relative_term        MPEP 2173.05(b): a term of degree with no standard for
                       measuring it in the limitation or its evidence quote
  exemplary_phrasing   MPEP 2173.05(d): "such as", "for example", "preferably"…
  functional_claiming  MPEP 2181 (A)(B)(C): "means/step for", or a nonce noun
                       with no structural modifier + functional language

undefined_term / contradicting_limitations / omission are left to the LLM
advisory (app.llm.definiteness_advisory) and never change the text.
`auto_fix` applies the rule-safe repairs (first "the X" -> "a X"; drop the
exemplary phrase); the rest goes to one reword call.
"""

from __future__ import annotations

import re

from patent_analyzer.agentic.elements import _GENERIC

_STOP = {"of", "for", "to", "that", "which", "configured", "comprising", "wherein", "having", "based", "in", "on", "by",
         "with", "and", "or", "from", "at", "into", "via", "using", "when", "where", "is", "are", "being", "such", "so",
         "as", "than", "includes", "including", "whereby", "the", "a", "an", "said", "each", "one", "any", "whether",
         "if", "while", "after", "before", "during", "between", "over", "under", "within", "without", "through",
         "according", "corresponding", "associated", "respectively", "thereof", "therein", "thereto", "further"}
_REF_WHITELIST = re.compile(
    r"^(?:(?:computer[- ]implemented\s+)?(?:method|system|apparatus|device|medium|process|composition|kit|assembly)"
    r"(?:\s+(?:of|according to)\s+claim)?|step|steps|invention|same|following|other|present|foregoing|preceding|above|"
    r"first|second|third|respective|corresponding|latter|former)\b", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|[,;:.()]")
_RELATIVE = re.compile(
    r"\b(about|approximately|substantially|essentially|relatively|similar|sufficient(?:ly)?|suitable|optimal|improved|"
    r"efficient|high|low|large|small|thin|thick|fast(?:er)?|slow(?:er)?|near|close|strong|weak|significant(?:ly)?|"
    r"effective(?:ly)?|adequate|appropriate|reasonable|minimal|maximal|enhanced|superior|better|optimized)\b", re.I)
_STANDARD = re.compile(r"\d|%|≥|≤|<|>|\bbetween\b|\brelative to\b|\bcompared (?:to|with)\b|\bthan\b|\bequal to\b", re.I)
_EXEMPLARY = re.compile(r"\b(such as|for example|for instance|e\.g\.|i\.e\.|preferably|optionally|or the like|etc\.?)", re.I)
_EX_PAREN = re.compile(r",?\s*\((?:e\.g\.|i\.e\.|for example|such as|for instance)[^)]*\)", re.I)
_EX_TAIL = re.compile(r"(,)?\s*(?:such as|for example|for instance|e\.g\.,?|i\.e\.,?)\s+[^;,.]*(?(1),?)", re.I)
_EX_WORD = re.compile(r"\b(?:preferably|optionally)\s+", re.I)
_EX_LIKE = re.compile(r",?\s*(?:or the like|and the like|etc\.?)", re.I)
_MEANS_FOR = re.compile(r"\b(means|step)\s+for\b", re.I)
_NONCE = {"mechanism", "module", "device", "unit", "component", "element", "member", "apparatus", "machine", "system",
          "means", "logic", "circuitry", "arrangement", "structure", "portion"}
_NONCE_RE = re.compile(
    r"\b(?:a|an|the|said|one or more|at least one)\s+((?:[A-Za-z][\w\-]*\s+){0,3}?)(" + "|".join(sorted(_NONCE)) +
    r")\s+(for\s+\w+ing|configured to|adapted to|operable to|arranged to|that\s+\w+s\b)", re.I)


def _singular(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ous")):
        return w[:-1]
    return w


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text or "")


def _participle(t: str) -> bool:
    return len(t) > 4 and (t.endswith("ed") or t.endswith("ing"))


def _np_after(tokens: list[str], i: int, allow_of: bool = True) -> tuple[list[str], int, list[str]]:
    """(np words, next index, weak np words) for the noun phrase starting at
    tokens[i]: up to a stop word / punctuation; a participle after the first
    word ends it unless another noun follows ('a value computed from' -> value;
    'a state embedding vector' -> all three). One 'of <NP>' continuation is
    included (a swarm of drones); 'of <gerund …>' (a method of routing packets)
    is not part of the phrase — its object comes back as a weak introduction."""
    out: list[str] = []
    j = i
    while j < len(tokens):
        t = tokens[j]
        low = t.lower()
        if not t[0].isalpha() or low in _STOP:
            break
        if out and _participle(low):
            nxt = tokens[j + 1].lower() if j + 1 < len(tokens) else ""
            if not nxt or not nxt[0].isalpha() or nxt in _STOP or _participle(nxt):
                break
        out.append(low)
        j += 1
    weak: list[str] = []
    if allow_of and out and j < len(tokens) and tokens[j].lower() == "of":
        sub, k, _ = _np_after(tokens, j + 1, allow_of=False)
        if sub and sub[0].endswith("ing"):
            weak = sub[1:]
            j = k
        elif sub:
            out = out + ["of"] + sub
            j = k
    return out, j, weak


def _key(np: list[str]) -> tuple:
    return tuple(_singular(w) for w in np)


def _new_event(intro: dict) -> int:
    ev = intro.setdefault(("_ev",), {"n": 0})
    ev["n"] += 1
    return ev["n"]


def _add(intro: dict, key: tuple, event: int | None) -> None:
    if not key:
        return
    e = intro.setdefault(key, {"events": set(), "weak": False})
    if event is None:
        e["weak"] = True
    else:
        e["events"].add(event)


def _register(intro: dict, np: list[str], weak: list[str] | None = None, strong: bool = True) -> None:
    """A strong introduction (a X / one or more X / a plurality of X) is one
    event; the phrase after its 'of' shares the event. Weak introductions
    (bare plurals, objects of a gerund) satisfy a later 'the X' but never make
    it ambiguous."""
    key = _key(np)
    if not key:
        return
    event = _new_event(intro) if strong else None
    _add(intro, key, event)
    if "of" in key:
        sub = key[key.index("of") + 1:]
        _add(intro, sub, event)
    if weak:
        _add(intro, _key(weak), None)


def _walk(text: str, intro: dict, on_ref=None) -> None:
    """One pass over `text`: every introduction is registered in `intro` as it
    is met; every 'the/said X' is handed to on_ref(span, np_words, intro) with
    the set as it stands at that point (a later introduction does not count)."""
    tokens = _tokens(text)
    lower = [t.lower() for t in tokens]
    i = 0
    while i < len(tokens):
        t = lower[i]
        if t in ("the", "said") and i + 1 < len(tokens):
            np, j, weak = _np_after(tokens, i + 1)
            if np and on_ref is not None and not _REF_WHITELIST.match(" ".join(np)):
                on_ref(f"{tokens[i]} {' '.join(np)}", np, intro)
            if weak:
                _add(intro, _key(weak), None)
            i = j if np else i + 1
            continue
        if t in ("a", "an") and i + 1 < len(tokens):
            np, j, weak = _np_after(tokens, i + 1)
            if np:
                _register(intro, np, weak)
                i = j
                continue
        if t in ("one", "at", "two") and " ".join(lower[i:i + 3]) in ("one or more", "at least one", "two or more"):
            np, j, weak = _np_after(tokens, i + 3)
            if np:
                _register(intro, np, weak)
                i = j
                continue
        if t in ("plurality", "set", "series", "number") and i + 1 < len(tokens) and lower[i + 1] == "of":
            np, j, weak = _np_after(tokens, i + 2)
            if np:
                _register(intro, np, weak)
                i = j
                continue
        # bare plural with no determiner before it (pulses, drones): weak introduction
        if (t[0].isalpha() and t not in _STOP and len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is", "ous"))
                and (i == 0 or lower[i - 1] in _STOP or not lower[i - 1][0].isalpha())):
            np, j, _ = _np_after(tokens, i, allow_of=False)
            if np:
                _register(intro, np, strong=False)
                i = j
                continue
        i += 1


def introduced_nps(text: str, intro: dict | None = None) -> dict:
    """Noun phrases introduced in `text` -> {"events": {ids of strong introductions}, "weak": bool}."""
    intro = intro if intro is not None else {}
    _walk(text, intro)
    return intro


def _contains(entry: tuple, key: tuple) -> bool:
    n = len(key)
    return any(entry[i:i + n] == key for i in range(len(entry) - n + 1))


def check_antecedent(text: str, intro: dict) -> list[dict]:
    """Flags for 'the/said X' with no, or an ambiguous, earlier introduction.
    `intro` (the parents' and earlier limitations' introductions) is updated in
    place as the text is walked, so call once per limitation in claim order."""
    flags = []

    def on_ref(span, np, cur):
        key = _key(np)
        hits = [(k, e) for k, e in cur.items() if k != ("_ev",) and _contains(k, key)]
        if not hits:
            base = [k for k, _ in cur.items() if k != ("_ev",) and key[-1] in k]
            note = (f"only '{' '.join(base[0])}' was introduced" if base else "no earlier introduction")
            flags.append({"category": "antecedent_basis", "span": span, "rule": "MPEP 2173.05(e)", "fixed": False,
                          "note": note, "kind": "no_antecedent"})
            return
        events = set()
        for _, e in hits:
            events |= e["events"]
        if len(events) >= 2:
            names = sorted({" ".join(k) for k, e in hits if e["events"]})[:3]
            flags.append({"category": "antecedent_basis", "span": span, "rule": "MPEP 2173.05(e)", "fixed": False,
                          "note": f"{len(events)} earlier introductions match ({', '.join(repr(n) for n in names)})",
                          "kind": "ambiguous_antecedent"})

    _walk(text, intro, on_ref)
    return flags


def check_relative(text: str, quotes: list[str] | None = None) -> list[dict]:
    m = _RELATIVE.search(text or "")
    if not m:
        return []
    pool = (text or "") + " " + " ".join(quotes or [])
    if _STANDARD.search(pool):
        return []
    return [{"category": "relative_term", "span": m.group(0), "rule": "MPEP 2173.05(b)", "fixed": False,
             "note": "no numeric / comparative standard in the limitation or its evidence quote"}]


def check_exemplary(text: str) -> list[dict]:
    m = _EXEMPLARY.search(text or "")
    if not m:
        return []
    return [{"category": "exemplary_phrasing", "span": m.group(0), "rule": "MPEP 2173.05(d)", "fixed": False,
             "note": "examples belong in the specification, not the claim"}]


def check_functional(text: str) -> list[dict]:
    flags = []
    m = _MEANS_FOR.search(text or "")
    if m:
        flags.append({"category": "functional_claiming", "span": m.group(0), "rule": "MPEP 2181 I(A)", "fixed": False,
                      "note": "'means/step for' invokes 112(f); the specification must disclose the structure/algorithm"})
    for m in _NONCE_RE.finditer(text or ""):
        mods = [w.lower() for w in m.group(1).split()]
        if all(w in _GENERIC for w in mods):
            flags.append({"category": "functional_claiming", "span": m.group(0), "rule": "MPEP 2181 I(A)-(C)", "fixed": False,
                          "note": f"generic placeholder '{m.group(2)}' with functional language and no structural modifier"})
    return flags


def _claim_texts(claim: dict) -> list[str]:
    return [claim.get("preamble") or ""] + [l.get("text") or "" for l in claim.get("limitations") or []]


def check(claim: dict, parents: list[dict] | None = None) -> list[dict]:
    """All rule flags for one claim, each tagged with the limitation's lid
    ('preamble' for the preamble). Parents (root first) seed the antecedent set."""
    intro: dict = {}
    for p in parents or []:
        for t in _claim_texts(p):
            introduced_nps(t, intro)
    flags = []
    pre_flags = check_antecedent(claim.get("preamble") or "", intro)
    for f in pre_flags:
        flags.append({"lid": "preamble", **f})
    for l in claim.get("limitations") or []:
        lid = l.get("lid", "")
        quotes = [b.get("evidence_quote", "") for b in l.get("basis") or []]
        for f in (check_antecedent(l.get("text") or "", intro) + check_relative(l.get("text") or "", quotes)
                  + check_exemplary(l.get("text") or "") + check_functional(l.get("text") or "")):
            flags.append({"lid": lid, **f})
    return flags


def _indefinite(np_text: str) -> str:
    words = np_text.split()
    first, last = (words[0] if words else ""), (words[-1] if words else "")
    if last and _singular(last) != last.lower():          # plural head: bare plural introduces it
        return np_text
    return ("an " if first[:1].lower() in "aeiou" else "a ") + np_text


def auto_fix(claim: dict, flags: list[dict]) -> tuple[dict, list[dict]]:
    """Rule-safe repairs in place: a 'the X' with no antecedent becomes 'a X'
    (its first occurrence); exemplary phrases are removed. Flags that were
    repaired get fixed=True; the others are left for the reword call."""
    lims = {l.get("lid"): l for l in claim.get("limitations") or []}
    for f in flags:
        target = claim if f.get("lid") == "preamble" else lims.get(f.get("lid"))
        if target is None:
            continue
        key = "preamble" if f.get("lid") == "preamble" else "text"
        text = target.get(key) or ""
        new = text
        if f["category"] == "antecedent_basis" and f.get("kind") == "no_antecedent":
            span = f["span"]
            np_text = span.split(" ", 1)[1] if " " in span else ""
            if np_text and re.search(r"\b" + re.escape(span) + r"\b", text, re.I):
                new = re.sub(r"\b" + re.escape(span) + r"\b", _indefinite(np_text), text, count=1, flags=re.I)
        elif f["category"] == "exemplary_phrasing":
            new = _EX_PAREN.sub("", text)
            new = _EX_TAIL.sub("", new)
            new = _EX_WORD.sub("", new)
            new = _EX_LIKE.sub("", new)
            new = " ".join(new.split()).strip(" ,;")
        if new != text and new.strip():
            target[key] = new
            f["fixed"] = True
            f["fixed_text"] = new
    return claim, flags


def open_flags(flags: list[dict]) -> list[dict]:
    return [f for f in flags if not f.get("fixed")]
