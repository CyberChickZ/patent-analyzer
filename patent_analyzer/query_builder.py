"""
Deterministic query builder — no LLM needed.

Takes delegation planning output (atoms + groups with anchor/expansion terms)
and generates Boolean queries for SerpAPI Google Patents and keyword queries
for Google Scholar.
"""

from itertools import product
from .config import MAX_OR_TERMS, MAX_QUERIES_PER_GROUP


def _quote(term: str) -> str:
    """Wrap a single term in double quotes for exact-match search."""
    return f'"{term}"'


def _or_group(terms: list) -> str:
    """Build: ("term1" OR "term2" OR "term3"), max MAX_OR_TERMS."""
    # Flatten any nested lists and ensure all items are strings
    flat = []
    for t in terms:
        if isinstance(t, list):
            flat.extend(str(x) for x in t if x)
        elif isinstance(t, str) and t.strip():
            flat.append(t)
    limited = flat[:MAX_OR_TERMS]
    if not limited:
        return ""
    inner = " OR ".join(_quote(t) for t in limited)
    return f"({inner})"


def _and_all(chunks: list[str]) -> str:
    """Join chunks with AND."""
    return " AND ".join(chunks)


def build_patent_queries(group: dict) -> list[str]:
    """
    Generate Boolean queries for Google Patents from a delegation group.

    Input group format:
    {
        "anchor_terms": [["term1", "term2"], ["term3"]],
        "expansion_terms": [["term4", "term5"], ["term6"]]
    }

    Output: list of Boolean query strings, max MAX_QUERIES_PER_GROUP.
    """
    # Handle variant key names from different LLMs (e.g. "anchor_terms[][]")
    anchors = group.get("anchor_terms") or group.get("anchor_terms[][]") or []
    expansions = group.get("expansion_terms") or group.get("expansion_terms[][]") or []
    # Flatten if LLM returned flat list of strings instead of list of lists
    if anchors and isinstance(anchors[0], str):
        anchors = [anchors]
    if expansions and isinstance(expansions[0], str):
        expansions = [expansions]
    # Filter empty sublists
    anchors = [a for a in anchors if a and any(a)]
    expansions = [e for e in expansions if e and any(e)]

    if not anchors:
        return []

    queries = set()

    # Strategy 1: anchor groups combined
    if len(anchors) >= 2:
        q = _and_all([_or_group(a) for a in anchors[:2]])
        queries.add(q)

    # Strategy 2: each anchor group + each expansion group
    for anc in anchors:
        for exp in expansions:
            q = _and_all([_or_group(anc), _or_group(exp)])
            queries.add(q)
            if len(queries) >= MAX_QUERIES_PER_GROUP:
                return list(queries)

    # Strategy 3: single anchor groups (broadest fallback)
    if not queries:
        for anc in anchors:
            queries.add(_or_group(anc))
            if len(queries) >= MAX_QUERIES_PER_GROUP:
                break

    return list(queries)[:MAX_QUERIES_PER_GROUP]


def build_scholar_queries(group: dict) -> list[str]:
    """
    Generate natural-language queries for Google Scholar.
    Scholar works better with plain keyword strings.
    """
    # Handle variant key names from different LLMs (e.g. "anchor_terms[][]")
    anchors = group.get("anchor_terms") or group.get("anchor_terms[][]") or []
    expansions = group.get("expansion_terms") or group.get("expansion_terms[][]") or []
    # Flatten if LLM returned flat list of strings instead of list of lists
    if anchors and isinstance(anchors[0], str):
        anchors = [anchors]
    if expansions and isinstance(expansions[0], str):
        expansions = [expansions]
    # Filter empty sublists
    anchors = [a for a in anchors if a and any(a)]
    expansions = [e for e in expansions if e and any(e)]

    queries = []

    # Combine first term from each anchor + expansion
    all_first_terms = []
    for terms_list in anchors + expansions:
        if terms_list:
            all_first_terms.append(terms_list[0])

    if all_first_terms:
        queries.append(" ".join(all_first_terms))

    # Each anchor group as its own query
    for anc in anchors:
        q = " ".join(anc[:2])  # top 2 terms
        if q not in queries:
            queries.append(q)

    return queries[:MAX_QUERIES_PER_GROUP]


def build_all_queries(delegation: dict) -> dict:
    """
    Convert full delegation planning output into queries.json format.

    Input: delegation dict with "groups" list from phase2.json
    Output: dict ready to save as queries.json
    """
    groups = delegation.get("groups", [])
    result = {"groups": []}

    for g in groups:
        entry = {
            "group_id": g.get("group_id", ""),
            "label": g.get("label", ""),
            "patent_queries": build_patent_queries(g),
            "paper_queries": build_scholar_queries(g),
        }
        result["groups"].append(entry)

    return result
