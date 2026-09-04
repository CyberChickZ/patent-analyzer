"""Shared helpers for the stage-level evals (FiNE-Patents adapters)."""

import json
import os
import random
import re
from pathlib import Path


def _effective(name: str, env: str) -> bool:
    """Whether a prompt-behaviour switch is ON for this run, asked of the code
    that actually implements it rather than of the environment.

    This has to be the effective value, not `os.getenv(env) == "1"`. A run file
    is keyed on model_tag(), and a file that exists is returned without calling
    the model at all, so a switch that is on by default but only tagged when
    set explicitly makes a run read another configuration's cached answer. That
    is how a lean deep-read gate once passed with "0 live, 0 cached". Both of
    these are currently on by default."""
    try:
        from app import llm
        return bool(getattr(llm, name)())
    except Exception:
        return os.getenv(env) == "1"


def model_tag() -> str:
    """Run-file suffix when a stage model is overridden (LLM_MODEL_<STAGE>) or a
    prompt-behaviour switch is on, so runs of different configurations never
    share a cached run file. Empty for the plain default.

    Still NOT covered: the prompt text itself. Editing an eval prompt without
    changing a model or a switch leaves the tag identical, so the stale run
    file has to be deleted by hand."""
    parts = [f"{st}-{os.getenv(f'LLM_MODEL_{st.upper()}')}" for st in ("extract", "screen", "eval", "idca")
             if os.getenv(f"LLM_MODEL_{st.upper()}")]
    if os.getenv("LLM_THINKING_LEVEL"):
        parts.append(f"think-{os.getenv('LLM_THINKING_LEVEL').lower()}")
    if _effective("_lean_eval", "EVAL_LEAN"):
        parts.append("lean")           # deep read without thinking and without the analysis prose
    if _effective("_bri_enabled", "EVAL_BRI"):
        parts.append("bri")            # claims construed under MPEP 2111 BRI
    cap = _dep_cap()
    if cap is not None:
        parts.append(f"dep{cap}")      # DRAFT_MAX_DEPENDENTS sweep: different caps, different claim sets
    return ("_" + "+".join(parts)) if parts else ""


def _dep_cap() -> int | None:
    """The dependent-claim cap when it is not the module default, else None.
    Same reason as _effective: a draft run file keyed only on the model would
    serve the 8-dependent claim set back to a run asking for 20."""
    try:
        from patent_analyzer.draft import avoid
        return avoid.max_dependents() if avoid.max_dependents() != avoid.MAX_DEPENDENTS else None
    except Exception:
        raw = os.getenv("DRAFT_MAX_DEPENDENTS")
        return int(raw) if raw and raw.isdigit() and int(raw) != 8 else None


def load_env_yaml(path: Path | None = None, override: bool = False) -> list[str]:
    """Load backend/.env.yaml (the Cloud Run env file: `KEY: "value"` lines)
    into os.environ for local evals; existing variables win unless override.
    Returns the names that were set. Nothing in the file is ever printed."""
    path = path or Path(__file__).parent.parent / ".env.yaml"
    if not path.exists():
        return []
    setv = []
    for line in path.read_text().splitlines():
        m = re.match(r'^([A-Z][A-Z0-9_]*):\s*"([^"]*)"', line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if override or not os.environ.get(k):
            os.environ[k] = v
            setv.append(k)
    return setv

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


_PUB_STRIP = re.compile(r"[\s/\-,.]")


def norm_pub(s: str) -> str:
    """Normalize a publication number across FiNE / BigQuery / SerpAPI spellings.

    'US 2008/025717 A1', 'US-2008025717-A1', 'US20080025717A1' -> 'US20080025717A1'.
    Year-prefixed US pre-grant numbers are widened to the 11-digit form.
    """
    if not s:
        return ""
    t = _PUB_STRIP.sub("", s.upper())
    m = re.match(r"^([A-Z]{2})(\d+)([A-Z]\d?)?$", t)
    if not m:
        return t
    cc, digits, kind = m.group(1), m.group(2), m.group(3) or ""
    if cc == "US" and len(digits) == 10 and digits[:2] in ("19", "20"):
        digits = digits[:4] + "0" + digits[4:]
    return f"{cc}{digits}{kind}"


def _app_meta(app_dir: Path) -> dict | None:
    try:
        meta = json.loads((app_dir / "metadata.json").read_text())
        bd = json.loads((app_dir / "breakdown.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    feats = [f.get("feature", "").strip() for f in (bd.get("breakdown") or [])]
    return {
        "app": app_dir.name,
        "split": meta.get("split"),
        "versions": meta.get("include_versions") or [],
        "language": bd.get("language"),
        "n_features": sum(1 for f in feats if len(f) > 15),
    }


def build_fixture(seed: int = 42, n_stage: int = 100, n_e2e: int = 50) -> dict:
    """Frozen app-id samples so every stage eval scores the same population.

    stage: EN, >=2 examiner features, rejected version present (any split).
    e2e:   test split, both rejected+granted versions, EN — paired runs.
    """
    metas = [m for m in (_app_meta(d) for d in sorted(DATA_DIR.iterdir()) if d.is_dir()) if m]
    stage_pool = [m["app"] for m in metas
                  if m["language"] == "EN" and m["n_features"] >= 2 and "rejected" in m["versions"]]
    e2e_pool = [m["app"] for m in metas
                if m["language"] == "EN" and m["split"] == "test"
                and {"rejected", "granted"} <= set(m["versions"])]
    rng = random.Random(seed)
    return {
        "seed": seed,
        "stage": sorted(rng.sample(stage_pool, min(n_stage, len(stage_pool)))),
        "e2e": sorted(rng.sample(e2e_pool, min(n_e2e, len(e2e_pool)))),
        "pool_sizes": {"stage": len(stage_pool), "e2e": len(e2e_pool)},
    }


def load_fixture(name: str = "fine_ids_seed42.json") -> dict:
    path = FIXTURE_DIR / name
    if not path.exists():
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_fixture(), indent=1))
    return json.loads(path.read_text())


def load_app(app: str) -> dict:
    d = DATA_DIR / app
    out = {"app": app}
    for name in ("rejected_patent", "granted_patent", "cited_patent", "breakdown", "metadata"):
        p = d / f"{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else None
    return out


def examiner_features(app_data: dict) -> list[str]:
    bd = (app_data.get("breakdown") or {}).get("breakdown") or []
    return [f.get("feature", "").strip() for f in bd if len(f.get("feature", "").strip()) > 15]


_RANGE = re.compile(r"^\[?0*(\d+)\]?(?:\s*(?:-|to)\s*\[?0*(\d+)\]?)?$")


def _expand_numbers(numbers) -> set[int]:
    out = set()
    for n in numbers or []:
        if isinstance(n, int):
            out.add(n)
            continue
        m = _RANGE.match(str(n).strip())
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        out.update(range(a, b + 1))
    return out


def cited_doc_label(app_data: dict) -> str | None:
    """Label (D1/D2/...) of the prior-art document that is cited_patent.

    Matched by normalized publication number against prior_art_documents;
    falls back to the only patent-type document when identifiers disagree.
    """
    bd = app_data.get("breakdown") or {}
    docs = bd.get("prior_art_documents") or []
    pub = norm_pub((app_data.get("cited_patent") or {}).get("publication_number") or "")
    if pub:
        for d in docs:
            if d.get("identifier") and norm_pub(d["identifier"]) == pub:
                return d.get("label")
    patents = [d for d in docs if d.get("type") == "patent"]
    if len(patents) == 1:
        return patents[0].get("label")
    return None


_GOLD_KINDS = ("paragraph", "claim", "abstract")


def _gold_passages(refs: list[dict]) -> set[tuple[str, int | None]]:
    out = set()
    for r in refs:
        loc = r.get("location") or {}
        kind = loc.get("reference_type")
        if kind == "abstract":
            out.add(("abstract", None))
        elif kind in ("paragraph", "claim"):
            out |= {(kind, n) for n in _expand_numbers(loc.get("numbers"))}
    return out


def breakdown_features(app_data: dict, doc_label: str | None = None) -> list[dict]:
    """Every examiner feature (FiNE alignment universe) with the gold passage
    set (kind, number) the examiner cited in the cited document. Passages
    follow FiNE's locate_cited_passages: only paragraph / claim / abstract
    references count; figure/page/component/section/other are dropped, so a
    feature cited only by figure has an empty set."""
    bd = (app_data.get("breakdown") or {}).get("breakdown") or []
    label = doc_label or cited_doc_label(app_data)
    out = []
    for f in bd:
        refs = [r for r in (f.get("prior_art_references") or []) if r.get("document") == label]
        passages = _gold_passages(refs)
        out.append({"feature": f.get("feature", "").strip(), "passages": passages,
                    "paragraphs": sorted(n for k, n in passages if k == "paragraph")})
    return out


def disclosed_features(app_data: dict, doc_label: str | None = None) -> list[dict]:
    """Examiner features the cited document discloses (gold for coverage):
    features with at least one paragraph/claim/abstract reference to the
    cited document. `passages` is the FiNE gold set of (kind, number);
    `paragraphs` keeps the 1-based paragraph indices for the legacy metric."""
    return [f for f in breakdown_features(app_data, doc_label)
            if len(f["feature"]) > 15 and f["passages"]]


def format_cited(patent: dict) -> str:
    """Render the cited prior art as numbered paragraphs (same shape FiNE
    feeds its models), so quote locations can be mapped back to indices."""
    parts = [f"# Title\n{patent.get('title') or 'N/A'}", f"# Abstract\n{patent.get('abstract') or 'N/A'}", "# Description"]
    for i, p in enumerate(patent.get("description") or []):
        p = p or ""
        if not re.match(r"^\[\d+\]", p):
            p = f"[{i + 1:04d}] {p}"
        parts.append(p)
    parts.append("# Claims")
    parts += [c for c in (patent.get("claims") or []) if c]
    return "\n".join(parts)
