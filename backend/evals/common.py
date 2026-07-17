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
